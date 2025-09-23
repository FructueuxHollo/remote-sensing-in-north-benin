import os
import pandas as pd
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm
import random


# --- CONFIGURATION ---
SAVI_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Originals\SAVI")
OUTPUT_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Tiled")
TILE_SIZES = [(640, 640), (1024, 1024)]
OVERLAP_RATIO = 0.25
IOU_THRESHOLD = 0.25

# NOUVEAU: Ratio désiré d'images de fond dans le dataset final
TARGET_BACKGROUND_RATIO = 0.15

# Convention CVAT: Person(0), Car(1), Bicycle(2), Cattle(3 et 4)
# Convention Finale: Person(0), Bicycle(1), Car(2), Cattle(3)
SAVI_CLASS_MAPPING = {
    0: 0,  # Person -> Person
    1: 2,  # Car (CVAT) -> Car (Final)
    2: 1,  # Bicycle (CVAT) -> Bicycle (Final)
    3: 3,  # Cattle (CVAT) -> Cattle (Final)
    4: 3   # Cattle (CVAT) -> Cattle (Final)
}

# --- FONCTIONS UTILITAIRES ---
def parse_metadata(file_path):
    metadata = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"AVERTISSEMENT: Fichier metadata.txt non trouvé dans {file_path.parent}")
        return None
    return metadata

def yolo_to_pixel_bbox(yolo_bbox, img_w, img_h):
    class_id, x_c, y_c, w, h = yolo_bbox
    abs_w, abs_h = w * img_w, h * img_h
    x_min, y_min = (x_c * img_w) - (abs_w / 2), (y_c * img_h) - (abs_h / 2)
    return [class_id, x_min, y_min, x_min + abs_w, y_min + abs_h]

# --- SCRIPT PRINCIPAL ---
def process_savi_dataset():
    if not SAVI_ROOT.is_dir():
        print(f"ERREUR: Le dossier source '{SAVI_ROOT}' n'a pas été trouvé.")
        return

    for tile_size in TILE_SIZES:
        tile_w, tile_h = tile_size
        output_dir_name = f"SAVI_tiled_{tile_w}x{tile_h}"
        output_dir = OUTPUT_ROOT / output_dir_name
        output_images_dir = output_dir / "images"
        output_labels_dir = output_dir / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        # Listes pour la phase de découverte
        positive_tiles_info = []
        background_tiles_info = []

        print(f"\n--- Phase 1: Découverte des tuiles pour la taille {tile_w}x{tile_h} ---")
        batch_folders = [d for d in (SAVI_ROOT / "images").iterdir() if d.is_dir()]

        for batch_folder in tqdm(batch_folders, desc="Découverte dans les lots"):
            metadata = parse_metadata(batch_folder / "metadata.txt")
            if not metadata: continue

            image_files = list(batch_folder.glob("*.jpg"))
            for image_path in image_files:
                original_image_num = image_path.stem
                label_path = SAVI_ROOT / "labels" / batch_folder.name / "labels" / "train" / (original_image_num + ".txt")

                with Image.open(image_path) as img:
                    img_w, img_h = img.size
                    original_bboxes_yolo = []
                    if label_path.exists():
                        with open(label_path, 'r') as f:
                            original_bboxes_yolo = [[int(p.split()[0])] + [float(coord) for coord in p.split()[1:]] for p in f if p.strip()]
                    
                    original_bboxes_pixel = [yolo_to_pixel_bbox(bbox, img_w, img_h) for bbox in original_bboxes_yolo]
                    stride_w, stride_h = int(tile_w * (1 - OVERLAP_RATIO)), int(tile_h * (1 - OVERLAP_RATIO))

                    for y in range(0, img_h, stride_h):
                        for x in range(0, img_w, stride_h):
                            tile_bbox = (x, y, min(x + tile_w, img_w), min(y + tile_h, img_h))
                            if (tile_bbox[2] - tile_bbox[0]) < tile_w * 0.5 or (tile_bbox[3] - tile_bbox[1]) < tile_h * 0.5: continue

                            new_annotations_yolo = []
                            for class_id_from_file, obj_x_min, obj_y_min, obj_x_max, obj_y_max in original_bboxes_pixel:
                                inter_x_min, inter_y_min = max(obj_x_min, tile_bbox[0]), max(obj_y_min, tile_bbox[1])
                                inter_x_max, inter_y_max = min(obj_x_max, tile_bbox[2]), min(obj_y_max, tile_bbox[3])
                                inter_w, inter_h = inter_x_max - inter_x_min, inter_y_max - inter_y_min

                                if inter_w > 0 and inter_h > 0:
                                    original_area = (obj_x_max - obj_x_min) * (obj_y_max - obj_y_min)
                                    if original_area > 0 and ((inter_w * inter_h) / original_area) > IOU_THRESHOLD:
                                        if class_id_from_file in SAVI_CLASS_MAPPING:
                                            final_class_id = SAVI_CLASS_MAPPING[class_id_from_file]
                                        new_x_c = ((inter_x_min - tile_bbox[0]) + inter_w / 2) / (tile_bbox[2] - tile_bbox[0])
                                        new_y_c = ((inter_y_min - tile_bbox[1]) + inter_h / 2) / (tile_bbox[3] - tile_bbox[1])
                                        new_w, new_h = inter_w / (tile_bbox[2] - tile_bbox[0]), inter_h / (tile_bbox[3] - tile_bbox[1])
                                        new_annotations_yolo.append(f"{final_class_id} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}")
                            
                            tile_info = {
                                "original_path": image_path,
                                "tile_bbox": tile_bbox,
                                "annotations": new_annotations_yolo,
                                "metadata": metadata,
                                "batch_name": batch_folder.name,
                                "original_num": original_image_num
                            }

                            if new_annotations_yolo:
                                positive_tiles_info.append(tile_info)
                            else:
                                background_tiles_info.append(tile_info)

        print(f"  -> Découverte terminée. Trouvé {len(positive_tiles_info)} tuiles avec objets et {len(background_tiles_info)} tuiles de fond potentielles.")

        # --- Phase 2: Échantillonnage et Écriture ---
        print(f"\n--- Phase 2: Échantillonnage et écriture des fichiers ---")

        # Calculer combien de tuiles de fond garder
        num_positive = len(positive_tiles_info)
        # Formule: N_neg / (N_pos + N_neg) = ratio -> N_neg = N_pos * ratio / (1 - ratio)
        num_background_to_keep = int(num_positive * (TARGET_BACKGROUND_RATIO / (1 - TARGET_BACKGROUND_RATIO)))
        
        # S'assurer de ne pas essayer d'échantillonner plus que ce qui est disponible
        num_background_to_keep = min(num_background_to_keep, len(background_tiles_info))
        
        print(f"  -> Objectif: {num_background_to_keep} tuiles de fond pour un ratio de {TARGET_BACKGROUND_RATIO*100:.1f}%.")
        
        # Échantillonner aléatoirement les tuiles de fond
        sampled_background_tiles = random.sample(background_tiles_info, num_background_to_keep)
        
        final_tiles_to_write = positive_tiles_info + sampled_background_tiles
        random.shuffle(final_tiles_to_write) # Mélanger pour une bonne répartition train/val/test future
        
        print(f"  -> Nombre total de tuiles à écrire : {len(final_tiles_to_write)}")

        all_metadata_rows = []
        for tile_info in tqdm(final_tiles_to_write, desc="Écriture des tuiles"):
            tile_bbox = tile_info["tile_bbox"]
            tile_y_min, tile_y_max = tile_bbox[1], tile_bbox[3]
            
            # Créer le nom de fichier final
            tile_filename_stem = f"SAVI_{tile_info['batch_name']}_{tile_info['original_num']}_{tile_y_min}_{tile_y_max}"
            
            # Sauvegarder l'image
            with Image.open(tile_info["original_path"]) as img:
                tile_image = img.crop(tile_bbox)
                tile_image.save(output_images_dir / f"{tile_filename_stem}.jpg")

            # Sauvegarder le fichier d'annotation (peut être vide)
            with open(output_labels_dir / f"{tile_filename_stem}.txt", 'w') as f_out:
                f_out.write("\n".join(tile_info["annotations"]))

            # Ajouter l'entrée pour le CSV
            metadata = tile_info["metadata"]
            row = {
                'id': tile_filename_stem,
                'angle': int(metadata.get('Angle', -1)),
                'altitude': int(metadata.get('Altitude', -1)),
                'meteo': metadata.get('Meteo', 'unknown'),
                'region': 'rural',
                'mode': metadata.get('Mode', 'unknown'),
                'y_start': tile_y_min,
                'y_end': tile_y_max
            }
            all_metadata_rows.append(row)
        
        # Créer et sauvegarder le fichier CSV
        if all_metadata_rows:
            df = pd.DataFrame(all_metadata_rows)
            csv_path = OUTPUT_ROOT / f"savi_metadata_{tile_w}x{tile_h}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  -> Fichier de métadonnées créé avec succès : {csv_path}")

if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(exist_ok=True)
    process_savi_dataset()
    print("\n--- Traitement du dataset SAVI terminé ! ---")
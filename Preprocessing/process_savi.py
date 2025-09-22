import os
import pandas as pd
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---

# Dossier racine de votre dataset SAVI original
SAVI_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Originals\SAVI")

# Dossier où seront stockés les résultats (datasets tuilés et CSV)
OUTPUT_ROOT = Path("D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Tiled")

# Tailles de tuiles à générer
TILE_SIZES = [(640, 640), (1024, 1024)]

# Paramètres de tiling (identiques au script précédent)
OVERLAP_RATIO = 0.25
IOU_THRESHOLD = 0.25

# --- FONCTIONS UTILITAIRES ---

def parse_metadata(file_path):
    """Lit un fichier metadata.txt et le retourne sous forme de dictionnaire."""
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
    """Convertit une bbox YOLO [class, x_c, y_c, w, h] en pixels [class, x_min, y_min, x_max, y_max]."""
    class_id, x_c, y_c, w, h = yolo_bbox
    abs_w = w * img_w
    abs_h = h * img_h
    x_min = (x_c * img_w) - (abs_w / 2)
    y_min = (y_c * img_h) - (abs_h / 2)
    x_max = x_min + abs_w
    y_max = y_min + abs_h
    return [class_id, x_min, y_min, x_max, y_max]

# --- SCRIPT PRINCIPAL ---

def process_savi_dataset():
    """
    Fonction principale pour tuiler le dataset SAVI et générer les fichiers de métadonnées.
    """
    if not SAVI_ROOT.is_dir():
        print(f"ERREUR: Le dossier source '{SAVI_ROOT}' n'a pas été trouvé.")
        return

    # Boucler sur chaque taille de tuile désirée (640x640, 1024x1024)
    for tile_size in TILE_SIZES:
        tile_w, tile_h = tile_size
        
        # Préparer les dossiers de sortie pour cette taille de tuile
        output_dir_name = f"SAVI_tiled_{tile_w}x{tile_h}"
        output_dir = OUTPUT_ROOT / output_dir_name
        output_images_dir = output_dir / "images"
        output_labels_dir = output_dir / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        # Initialiser la liste qui contiendra les données pour le CSV
        all_metadata_rows = []

        print(f"\n--- Début du traitement pour la taille de tuile {tile_w}x{tile_h} ---")

        # Trouver tous les dossiers de lots (ex: Bassa_01)
        batch_folders = [d for d in (SAVI_ROOT / "images").iterdir() if d.is_dir()]

        for batch_folder in tqdm(batch_folders, desc="Traitement des lots"):
            batch_name = batch_folder.name
            
            # 1. Charger les métadonnées pour ce lot
            metadata = parse_metadata(batch_folder / "metadata.txt")
            if not metadata:
                continue

            # 2. Parcourir toutes les images du lot
            image_files = list(batch_folder.glob("*.jpg"))
            for image_path in image_files:
                original_image_num = image_path.stem
                
                # Construire le chemin complexe vers le fichier d'annotation
                label_path = SAVI_ROOT / "labels" / batch_name / "labels" / "train" / (original_image_num + ".txt")

                with Image.open(image_path) as img:
                    img_w, img_h = img.size
                    
                    original_bboxes_yolo = []
                    if label_path.exists():
                        with open(label_path, 'r') as f:
                            for line in f:
                                if line.strip():
                                    parts = line.strip().split()
                                    original_bboxes_yolo.append([int(parts[0])] + [float(p) for p in parts[1:]])
                    
                    original_bboxes_pixel = [yolo_to_pixel_bbox(bbox, img_w, img_h) for bbox in original_bboxes_yolo]

                    stride_w = int(tile_w * (1 - OVERLAP_RATIO))
                    stride_h = int(tile_h * (1 - OVERLAP_RATIO))

                    # 3. Appliquer la logique de tiling
                    for y in range(0, img_h, stride_h):
                        for x in range(0, img_w, stride_w):
                            tile_x_min, tile_y_min = x, y
                            tile_x_max, tile_y_max = min(x + tile_w, img_w), min(y + tile_h, img_h)
                            
                            current_tile_w = tile_x_max - tile_x_min
                            current_tile_h = tile_y_max - tile_y_min

                            if current_tile_w < tile_w * 0.5 or current_tile_h < tile_h * 0.5:
                                continue

                            new_annotations_yolo = []
                            for class_id, obj_x_min, obj_y_min, obj_x_max, obj_y_max in original_bboxes_pixel:
                                inter_x_min, inter_y_min = max(obj_x_min, tile_x_min), max(obj_y_min, tile_y_min)
                                inter_x_max, inter_y_max = min(obj_x_max, tile_x_max), min(obj_y_max, tile_y_max)
                                
                                inter_w, inter_h = inter_x_max - inter_x_min, inter_y_max - inter_y_min

                                if inter_w > 0 and inter_h > 0:
                                    original_area = (obj_x_max - obj_x_min) * (obj_y_max - obj_y_min)
                                    if original_area > 0 and ((inter_w * inter_h) / original_area) > IOU_THRESHOLD:
                                        new_x_c = ((inter_x_min - tile_x_min) + inter_w / 2) / current_tile_w
                                        new_y_c = ((inter_y_min - tile_y_min) + inter_h / 2) / current_tile_h
                                        new_w, new_h = inter_w / current_tile_w, inter_h / current_tile_h
                                        new_annotations_yolo.append(f"{class_id} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}")

                            # 4. Sauvegarder la tuile et l'annotation si elle contient des objets
                            if new_annotations_yolo:
                                # Créer le nom de fichier final
                                tile_filename_stem = f"SAVI_{batch_name}_{original_image_num}_{tile_y_min}_{tile_y_max}"
                                
                                # Sauvegarder l'image tuilée
                                tile_image = img.crop((tile_x_min, tile_y_min, tile_x_max, tile_y_max))
                                tile_image.save(output_images_dir / f"{tile_filename_stem}.jpg")
                                
                                # Sauvegarder le fichier d'annotation
                                with open(output_labels_dir / f"{tile_filename_stem}.txt", 'w') as f_out:
                                    f_out.write("\n".join(new_annotations_yolo))

                                # 5. Créer l'entrée pour le fichier CSV
                                row = {
                                    'id': tile_filename_stem,
                                    'angle': int(metadata.get('Angle', -1)),
                                    'altitude': int(metadata.get('Altitude', -1)),
                                    'meteo': metadata.get('Meteo', 'unknown'),
                                    'region': 'rural', # Valeur fixe comme demandé
                                    'mode': metadata.get('Mode', 'unknown'),
                                    'y_start': tile_y_min,
                                    'y_end': tile_y_max
                                }
                                all_metadata_rows.append(row)
        
        # 6. Créer et sauvegarder le fichier CSV pour cette taille de tuile
        if all_metadata_rows:
            df = pd.DataFrame(all_metadata_rows)
            csv_path = OUTPUT_ROOT / f"savi_metadata_{tile_w}x{tile_h}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  -> Fichier de métadonnées créé avec succès : {csv_path}")
        else:
            print("  -> Aucune annotation trouvée, aucun fichier CSV n'a été créé.")

if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(exist_ok=True)
    process_savi_dataset()
    print("\n--- Traitement du dataset SAVI terminé ! ---")
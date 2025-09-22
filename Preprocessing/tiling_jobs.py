import os
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---

# Dossier racine contenant tous les datasets convertis
INPUT_DATASETS_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Converted")

# Dossier où seront stockés tous les nouveaux datasets tuilés
OUTPUT_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Tiled")

# Définition des tâches de tiling.
# Format: { "nom_du_dossier_source": [ (tile_w, tile_h), (autre_tile_w, autre_tile_h), ... ] }
# Les datasets non listés ici seront ignorés.
TILING_JOBS = {
    "POP": [(640, 640)],
    "VisDrone": [(640, 640), (1024, 1024)]
}

# Pourcentage de chevauchement entre les tuiles
OVERLAP_RATIO = 0.25

# Seuil pour garder une annotation dans une tuile.
# Si la surface de l'objet visible dans la tuile est < X% de sa surface originale, on l'ignore.
# Cela évite de garder des fragments d'objets inutiles.
IOU_THRESHOLD = 0.25 

# --- SCRIPT PRINCIPAL ---

def yolo_to_pixel_bbox(yolo_bbox, img_w, img_h):
    """Convertit une bbox YOLO [class, x_c, y_c, w, h] en coordonnées pixel [class, x_min, y_min, x_max, y_max]."""
    class_id, x_c, y_c, w, h = yolo_bbox
    abs_w = w * img_w
    abs_h = h * img_h
    abs_x_c = x_c * img_w
    abs_y_c = y_c * img_h
    x_min = abs_x_c - (abs_w / 2)
    y_min = abs_y_c - (abs_h / 2)
    x_max = x_min + abs_w
    y_max = y_min + abs_h
    return [class_id, x_min, y_min, x_max, y_max]

def tile_dataset(source_dataset_name, tile_size):
    """
    Fonction principale pour tuiler un dataset entier pour une taille de tuile donnée.
    """
    tile_w, tile_h = tile_size
    source_dir = INPUT_DATASETS_ROOT / source_dataset_name
    
    # Créer un nom de dossier de sortie descriptif
    output_dir_name = f"{source_dataset_name}_tiled_{tile_w}x{tile_h}"
    output_dir = OUTPUT_ROOT / output_dir_name
    
    print(f"\n--- Début du tiling pour '{source_dataset_name}' en tuiles de {tile_w}x{tile_h} ---")
    print(f"  -> Données sources : {source_dir}")
    print(f"  -> Données de sortie : {output_dir}")
    
    if not source_dir.is_dir():
        print(f"ERREUR: Le dossier source '{source_dir}' n'existe pas. Tâche ignorée.")
        return

    # Parcourir les splits (train, val, test)
    for split in ["train", "val", "test"]:
        source_images_dir = source_dir / "images" / split
        source_labels_dir = source_dir / "labels" / split
        
        if not source_images_dir.is_dir():
            print(f"  -> Pas de dossier '{split}' dans les images. Sous-ensemble ignoré.")
            continue
            
        output_images_dir = output_dir / "images" / split
        output_labels_dir = output_dir / "labels" / split
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(source_images_dir.glob("*.jpg"))
        
        for image_path in tqdm(image_files, desc=f"Tiling {split}"):
            label_path = source_labels_dir / (image_path.stem + ".txt")
            
            with Image.open(image_path) as img:
                img_w, img_h = img.size
                
                # Charger les annotations originales s'il y en a
                original_bboxes_yolo = []
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                parts = line.strip().split()
                                original_bboxes_yolo.append([int(parts[0])] + [float(p) for p in parts[1:]])
                
                # Convertir toutes les bboxes YOLO en pixels pour faciliter les calculs
                original_bboxes_pixel = [yolo_to_pixel_bbox(bbox, img_w, img_h) for bbox in original_bboxes_yolo]

                # Calculer le pas (stride) pour le découpage
                stride_w = int(tile_w * (1 - OVERLAP_RATIO))
                stride_h = int(tile_h * (1 - OVERLAP_RATIO))

                # Parcourir l'image pour créer des tuiles
                for y in range(0, img_h, stride_h):
                    for x in range(0, img_w, stride_w):
                        # Coordonnées de la tuile dans l'image originale
                        tile_x_min = x
                        tile_y_min = y
                        tile_x_max = min(x + tile_w, img_w)
                        tile_y_max = min(y + tile_h, img_h)
                        
                        # Dimensions réelles de la tuile (peut être plus petite sur les bords)
                        current_tile_w = tile_x_max - tile_x_min
                        current_tile_h = tile_y_max - tile_y_min

                        # Si la tuile est trop petite (artefact sur les bords), on l'ignore
                        if current_tile_w < tile_w * 0.5 or current_tile_h < tile_h * 0.5:
                            continue

                        new_annotations_yolo = []
                        
                        # Vérifier chaque objet original
                        for class_id, obj_x_min, obj_y_min, obj_x_max, obj_y_max in original_bboxes_pixel:
                            # Calculer l'intersection entre l'objet et la tuile
                            inter_x_min = max(obj_x_min, tile_x_min)
                            inter_y_min = max(obj_y_min, tile_y_min)
                            inter_x_max = min(obj_x_max, tile_x_max)
                            inter_y_max = min(obj_y_max, tile_y_max)
                            
                            inter_w = inter_x_max - inter_x_min
                            inter_h = inter_y_max - inter_y_min

                            # S'il y a une intersection valide
                            if inter_w > 0 and inter_h > 0:
                                original_area = (obj_x_max - obj_x_min) * (obj_y_max - obj_y_min)
                                intersection_area = inter_w * inter_h
                                
                                # Appliquer le seuil pour éviter les fragments
                                if original_area > 0 and (intersection_area / original_area) > IOU_THRESHOLD:
                                    # Recalculer les coordonnées relatives à la tuile
                                    new_x_min = inter_x_min - tile_x_min
                                    new_y_min = inter_y_min - tile_y_min
                                    
                                    # Convertir en format YOLO pour la tuile
                                    new_x_c = (new_x_min + inter_w / 2) / current_tile_w
                                    new_y_c = (new_y_min + inter_h / 2) / current_tile_h
                                    new_w = inter_w / current_tile_w
                                    new_h = inter_h / current_tile_h
                                    
                                    new_annotations_yolo.append(f"{class_id} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}")

                        # Si la tuile contient au moins un objet, on la sauvegarde
                        if new_annotations_yolo:
                            # Découper l'image
                            tile_image = img.crop((tile_x_min, tile_y_min, tile_x_max, tile_y_max))
                            
                            # Créer un nom de fichier unique pour la tuile
                            tile_filename_stem = f"{image_path.stem}__{tile_x_min}_{tile_y_min}"
                            
                            # Sauvegarder la nouvelle image et le nouveau fichier d'annotation
                            tile_image.save(output_images_dir / f"{tile_filename_stem}.jpg")
                            
                            with open(output_labels_dir / f"{tile_filename_stem}.txt", 'w') as f_out:
                                f_out.write("\n".join(new_annotations_yolo))

if __name__ == "__main__":
    # Créer le dossier de sortie principal s'il n'existe pas
    OUTPUT_ROOT.mkdir(exist_ok=True)
    
    # Lancer toutes les tâches définies dans la configuration
    for dataset_name, tile_sizes in TILING_JOBS.items():
        for size in tile_sizes:
            tile_dataset(dataset_name, size)
    
    print("\n--- Tiling de tous les datasets terminé ! ---")
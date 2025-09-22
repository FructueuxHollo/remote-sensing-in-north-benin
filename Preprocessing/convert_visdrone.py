import os
from pathlib import Path
import shutil
from PIL import Image # Pillow est nécessaire pour obtenir les dimensions des images

# --- CONFIGURATION ---
VISDRONE_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Originals\VisDrone") # Chemin vers le dossier racine de VisDrone
OUTPUT_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Converted\VisDrone") # Chemin où le nouveau dataset sera créé

# Dictionnaire de mapping des classes VisDrone vers notre convention
# Notre convention: Person (0), Bicycle(1), Car(2), Cattle(3)
# VisDrone categories:
# ignored regions(0), pedestrian(1), people(2), bicycle(3), car(4), 
# van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), others(11)

CLASS_MAPPING = {
    1: 0,  # pedestrian -> Person
    2: 0,  # people -> Person
    3: 1,  # bicycle -> Bicycle
    4: 2,  # car -> Car
    5: 2,  # van -> Car
    6: 2,  # truck -> Car
    7: 1,  # tricycle -> Bicycle
    8: 1,  # awning-tricycle -> Bicycle
    9: 2,  # bus -> Car
    10: 1, # motor -> Bicycle
}

# Classes à ignorer complètement dans VisDrone
CLASSES_TO_IGNORE = {0, 11} # ignored regions, others

# --- FONCTION DE CONVERSION BBOX (identique à celle pour POP) ---
def convert_coco_to_yolo(x_min, y_min, width, height, img_width, img_height):
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return x_center, y_center, norm_width, norm_height

# --- SCRIPT DE CONVERSION ---
def convert_visdrone_dataset():
    """
    Script principal pour convertir le dataset VisDrone au format YOLO.
    """
    print("Début de la conversion du dataset VisDrone...")
    
    # Créer la structure de dossiers de destination
    output_labels_dir = OUTPUT_ROOT / "labels"
    output_images_dir = OUTPUT_ROOT / "images"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Note: Le README parle de 'training data', 'validation data'.
    # Les dossiers sont souvent nommés 'VisDrone2019-DET-train', etc.
    # Nous allons chercher les splits train/val/test dans le dossier source.
    # J'utilise les noms de dossiers que vous avez mentionnés.
    splits = ["train", "val", "test"] 
    
    for split in splits:
        print(f"\n--- Traitement du sous-ensemble : {split} ---")
        
        source_annotations_dir = VISDRONE_ROOT / "labels" # VisDrone utilise "annotations"
        source_images_dir = VISDRONE_ROOT / "images"
        
        # Ajustement pour la structure de dossier de VisDrone qui met le split dans le nom
        # Ex: VisDrone2019-DET-train/
        # Si votre structure est /images/train, /annotations/train, laissez comme c'est.
        # Sinon, ajustez ici. On part du principe que c'est /annotations/train etc.
        source_split_annotations = source_annotations_dir / split
        source_split_images = source_images_dir / split
        
        output_split_labels_dir = output_labels_dir / split
        output_split_images_dir = output_images_dir / split
        output_split_labels_dir.mkdir(exist_ok=True)

        if not source_split_annotations.is_dir():
            print(f"AVERTISSEMENT: Dossier d'annotations '{source_split_annotations}' non trouvé. Sous-ensemble ignoré.")
            continue

        # Copier les images d'abord
        print(f"Copie des images pour le sous-ensemble {split}...")
        if source_split_images.is_dir():
            shutil.copytree(source_split_images, output_split_images_dir, dirs_exist_ok=True)
        else:
            print(f"AVERTISSEMET: Dossier d'images '{source_split_images}' non trouvé.")
            continue
        
        # Dictionnaire pour stocker les dimensions des images pour éviter de les lire plusieurs fois
        image_dims = {}
        
        label_files = list(source_split_annotations.glob("*.txt"))
        print(f"Traitement de {len(label_files)} fichiers d'annotation...")

        for label_file_path in label_files:
            image_name = label_file_path.stem + ".jpg"
            image_path = output_split_images_dir / image_name
            
            # Obtenir les dimensions de l'image (une seule fois)
            if image_path not in image_dims:
                try:
                    with Image.open(image_path) as img:
                        image_dims[image_path] = img.size # (width, height)
                except FileNotFoundError:
                    print(f"AVERTISSEMENT: Image '{image_path}' non trouvée. Fichier d'annotation ignoré.")
                    continue
            
            img_width, img_height = image_dims[image_path]
            
            yolo_annotations = []
            with open(label_file_path, 'r') as f_in:
                for line in f_in:
                    if line.strip():
                        try:
                            cleaned_line = line.strip().rstrip(',')
                            parts = [int(p) for p in cleaned_line.split(',')]
                            
                            # Vérifier que la ligne a bien le bon nombre de colonnes
                            if len(parts) < 8:
                                continue

                            bbox_left, bbox_top, bbox_width, bbox_height, score, category_id, _, _ = parts
                            
                            if score == 0:
                                continue
                            if category_id in CLASSES_TO_IGNORE:
                                continue
                            
                            if category_id not in CLASS_MAPPING:
                                continue 
                            
                            new_class_id = CLASS_MAPPING[category_id]
                            
                            yolo_bbox = convert_coco_to_yolo(bbox_left, bbox_top, bbox_width, bbox_height, img_width, img_height)
                            
                            yolo_line = f"{new_class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                            yolo_annotations.append(yolo_line)
                        
                        except ValueError:
                            print(f"AVERTISSEMENT: Ligne mal formée ou non-entière dans '{label_file_path.name}': '{line.strip()}'. Ligne ignorée.")
                            continue
            
            # Écrire le nouveau fichier d'annotation
            if yolo_annotations:
                output_label_path = output_split_labels_dir / label_file_path.name
                with open(output_label_path, 'w') as f_out:
                    f_out.write("\n".join(yolo_annotations))

    print(f"\nConversion de VisDrone terminée.")
    print(f"Les données converties sont disponibles dans : '{OUTPUT_ROOT}'")

if __name__ == "__main__":
    # Assurez-vous d'avoir installé Pillow: pip install Pillow
    convert_visdrone_dataset()
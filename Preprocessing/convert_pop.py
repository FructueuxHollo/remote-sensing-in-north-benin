import os
import json
from pathlib import Path
import shutil
from collections import defaultdict

# --- CONFIGURATION ---
POP_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Originals\POP")  # Chemin vers le dossier racine de POP
OUTPUT_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Converted\POP") # Chemin où le nouveau dataset sera créé

# Dictionnaire de mapping pour POP
# POP n'a qu'une seule classe (Person) avec l'ID 1.
# On la mappe à notre classe Person qui a l'ID 0.
CLASS_MAPPING = {
    1: 0  # Person -> Person
}

# --- FONCTION DE CONVERSION BBOX ---
def convert_coco_to_yolo(x_min, y_min, width, height, img_width, img_height):
    """
    Convertit une bounding box du format COCO [x_min, y_min, w, h]
    au format YOLO normalisé [x_center, y_center, w, h].
    """
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return x_center, y_center, norm_width, norm_height

# --- SCRIPT DE CONVERSION ---
def convert_pop_dataset():
    """
    Script principal pour convertir le dataset POP du format COCO (JSON) au format YOLO (TXT).
    """
    print("Début de la conversion du dataset POP...")

    # Vérifier si le dossier source existe
    if not POP_ROOT.is_dir():
        print(f"ERREUR: Le dossier source '{POP_ROOT}' n'a pas été trouvé.")
        return

    # Créer la structure de dossiers de destination
    output_labels_dir = OUTPUT_ROOT / "labels"
    output_images_dir = OUTPUT_ROOT / "images"
    
    print(f"Création de la structure de destination dans '{OUTPUT_ROOT}'...")
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Parcourir les sous-ensembles (train, val, test)
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\n--- Traitement du sous-ensemble : {split} ---")
        
        # Définir les chemins source et destination
        source_json_path = POP_ROOT / "labels" / split / f"{split}.json"
        source_images_dir = POP_ROOT / "images" / split
        
        output_split_labels_dir = output_labels_dir / split
        output_split_images_dir = output_images_dir / split
        output_split_labels_dir.mkdir(exist_ok=True)
        # S'assurer que le dossier d'images de destination est vide avant la copie
        if output_split_images_dir.exists():
            shutil.rmtree(output_split_images_dir)
        
        if not source_json_path.is_file():
            print(f"AVERTISSEMENT: Fichier JSON '{source_json_path}' non trouvé. Sous-ensemble ignoré.")
            continue

        # Copier les images
        print(f"Copie des images pour le sous-ensemble {split}...")
        if source_images_dir.is_dir():
            shutil.copytree(source_images_dir, output_split_images_dir)
        else:
            print(f"AVERTISSEMENT: Le dossier d'images '{source_images_dir}' n'existe pas.")
            continue

        print("Renommage des extensions d'images en minuscules...")
        files_renamed_count = 0
        for jpg_file in output_split_images_dir.rglob('*.JPG'):
            new_file_path = jpg_file.with_suffix('.jpg')
            jpg_file.rename(new_file_path)
            files_renamed_count += 1
        
        if files_renamed_count > 0:
            print(f"{files_renamed_count} fichiers ont été renommés en .jpg.")

        # Charger le fichier JSON
        with open(source_json_path, 'r') as f:
            data = json.load(f)

        # Créer des mappings pour un accès rapide
        # image_id -> {file_name, width, height}
        images_info = {img['id']: img for img in data['images']}
        
        # image_id -> [liste des annotations]
        annotations_by_image = defaultdict(list)
        for ann in data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
            
        print(f"Traitement de {len(images_info)} images et {len(data['annotations'])} annotations...")

        # Parcourir chaque image et créer le fichier d'annotation YOLO correspondant
        for image_id, annotations in annotations_by_image.items():
            image_details = images_info.get(image_id)
            if not image_details:
                continue

            img_width = image_details['width']
            img_height = image_details['height']
            
            # Le nom du fichier txt doit correspondre au nom de l'image (sans extension)
            image_details['file_name'] = Path(image_details['file_name']).stem + '.jpg'
            file_name_stem = Path(image_details['file_name']).stem
            output_label_path = output_split_labels_dir / f"{file_name_stem}.txt"
            
            yolo_annotations = []
            for ann in annotations:
                coco_bbox = ann['bbox']
                category_id = ann['category_id']
                
                # Mapper la classe
                if category_id not in CLASS_MAPPING:
                    continue
                
                new_class_id = CLASS_MAPPING[category_id]
                
                # Convertir la bbox
                x_min, y_min, width, height = coco_bbox
                yolo_bbox = convert_coco_to_yolo(x_min, y_min, width, height, img_width, img_height)
                
                # Formater pour l'écriture
                yolo_line = f"{new_class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                yolo_annotations.append(yolo_line)

            # Écrire le fichier d'annotation
            if yolo_annotations:
                with open(output_label_path, 'w') as f_out:
                    f_out.write("\n".join(yolo_annotations))

    print(f"\nConversion de POP terminée.")
    print(f"Les données converties sont disponibles dans : '{OUTPUT_ROOT}'")

if __name__ == "__main__":
    convert_pop_dataset()
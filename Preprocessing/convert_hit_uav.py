import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---
# Modifiez ces chemins selon votre structure de dossiers
HIT_UAV_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Originals\HIT-UAV")  # Chemin vers le dossier racine de HIT-UAV
OUTPUT_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Converted\HIT-UAV") # Chemin où le nouveau dataset sera créé

# Dictionnaire de mapping : {ancien_indice_classe: nouvel_indice_classe}
# Selon votre convention : Person (0), Bicycle(1), Car(2)
# HIT-UAV: Person(0), Car(1), Bicycle(2)
CLASS_MAPPING = {
    0: 0,  # Person -> Person
    1: 2,  # Car -> Car
    2: 1   # Bicycle -> Bicycle
}

# Indices des classes à supprimer complètement
CLASSES_TO_IGNORE = {3, 4} # OtherVehicle et DontCare

# --- SCRIPT DE CONVERSION ---

def convert_hit_uav_labels():
    """
    Script principal pour convertir les annotations du dataset HIT-UAV.
    """
    print("Début de la conversion des annotations de HIT-UAV...")

    # Vérifier si le dossier source existe
    source_labels_dir = HIT_UAV_ROOT / "labels"
    if not source_labels_dir.is_dir():
        print(f"ERREUR: Le dossier des labels '{source_labels_dir}' n'a pas été trouvé.")
        print("Veuillez vérifier le chemin dans la variable HIT_UAV_ROOT.")
        return

    # Créer la structure de dossiers de destination
    output_labels_dir = OUTPUT_ROOT / "labels"
    output_images_dir = OUTPUT_ROOT / "images"
    
    print(f"Création de la structure de destination dans '{OUTPUT_ROOT}'...")
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Parcourir les sous-ensembles (train, val, test)
    splits = ["train", "val", "test"]
    total_files_processed = 0

    for split in splits:
        print(f"\n--- Traitement du sous-ensemble : {split} ---")
        
        source_split_dir = source_labels_dir / split
        output_split_dir = output_labels_dir / split
        output_split_dir.mkdir(exist_ok=True)

        # Vérifier si le dossier du split source existe
        if not source_split_dir.is_dir():
            print(f"AVERTISSEMENT: Le dossier '{source_split_dir}' n'existe pas, il est ignoré.")
            continue
            
        # Copier les images correspondantes
        print(f"Copie des images pour le sous-ensemble {split}...")
        source_images_split_dir = HIT_UAV_ROOT / "images" / split
        output_images_split_dir = output_images_dir / split
        if source_images_split_dir.is_dir():
            shutil.copytree(source_images_split_dir, output_images_split_dir, dirs_exist_ok=True)
        else:
             print(f"AVERTISSEMENT: Le dossier d'images '{source_images_split_dir}' n'existe pas.")


        # Lister tous les fichiers d'annotation dans le dossier source du split
        label_files = list(source_split_dir.glob("*.txt"))
        if not label_files:
            print("Aucun fichier d'annotation trouvé.")
            continue
            
        print(f"{len(label_files)} fichiers d'annotation à traiter...")
        
        for label_file_path in label_files:
            new_annotations = []
            
            with open(label_file_path, 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    try:
                        source_class_id = int(parts[0])
                        
                        # Étape 1: Filtrer les classes non désirées
                        if source_class_id in CLASSES_TO_IGNORE:
                            continue
                        
                        # Étape 2: Mapper les classes restantes
                        if source_class_id in CLASS_MAPPING:
                            target_class_id = CLASS_MAPPING[source_class_id]
                            # Reconstruire la ligne avec le nouvel ID de classe
                            new_line = f"{target_class_id} {' '.join(parts[1:])}"
                            new_annotations.append(new_line)
                        else:
                            # Optionnel: Avertir si une classe inattendue est trouvée
                            print(f"AVERTISSEMENT: ID de classe inattendu '{source_class_id}' trouvé dans le fichier '{label_file_path.name}'. Ligne ignorée.")

                    except (ValueError, IndexError) as e:
                        print(f"ERREUR: Ligne mal formée dans '{label_file_path.name}': '{line.strip()}'. Erreur: {e}")

            # Écrire le nouveau fichier d'annotation seulement s'il contient des annotations utiles
            if new_annotations:
                output_file_path = output_split_dir / label_file_path.name
                with open(output_file_path, 'w') as f_out:
                    f_out.write("\n".join(new_annotations))
            
            total_files_processed += 1

    print(f"\nConversion terminée. {total_files_processed} fichiers d'annotation traités.")
    print(f"Les données converties sont disponibles dans : '{OUTPUT_ROOT}'")

if __name__ == "__main__":
    convert_hit_uav_labels()
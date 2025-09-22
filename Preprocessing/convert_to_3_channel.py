import os
from pathlib import Path
from PIL import Image

# --- Prérequis ---
# Assurez-vous d'avoir installé Pillow et tqdm :
# pip install Pillow tqdm

from tqdm import tqdm

# --- CONFIGURATION ---
# Liste des dossiers racines des datasets à traiter
# Ce sont les datasets que vous avez créés aux étapes précédentes
DATASET_ROOTS = [
    Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Converted\HIT-UAV"),
    Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Converted\POP")
]

# --- SCRIPT DE CONVERSION ---

def convert_grayscale_to_rgb(dataset_paths):
    """
    Parcourt les datasets spécifiés et convertit toutes les images
    en niveaux de gris (1 canal) en images RGB (3 canaux) en dupliquant le canal.
    """
    for root_path in dataset_paths:
        if not root_path.is_dir():
            print(f"AVERTISSEMENT: Le dossier '{root_path}' n'existe pas. Il est ignoré.")
            continue

        print(f"\n--- Traitement du dataset : {root_path} ---")
        
        images_dir = root_path / "images"
        if not images_dir.is_dir():
            print(f"  -> Le sous-dossier 'images' n'a pas été trouvé. Dataset ignoré.")
            continue
            
        # Récupérer la liste de toutes les images (.jpg, .png, etc.) dans les sous-dossiers
        image_files = list(images_dir.rglob('*.jpg')) + \
                      list(images_dir.rglob('*.jpeg')) + \
                      list(images_dir.rglob('*.png'))
        
        if not image_files:
            print("  -> Aucune image trouvée dans ce dataset.")
            continue
            
        converted_count = 0
        skipped_count = 0
        
        # Utiliser tqdm pour afficher une barre de progression
        for image_path in tqdm(image_files, desc=f"Conversion de {root_path.name}"):
            try:
                with Image.open(image_path) as img:
                    # 'L' est le mode pour les images en niveaux de gris (Luminance)
                    if img.mode != 'RGB':
                        # La méthode .convert('RGB') duplique le canal 'L' dans R, G, et B
                        rgb_img = img.convert('RGB')
                        # Sauvegarder l'image, écrasant l'ancienne version
                        rgb_img.save(image_path)
                        converted_count += 1
                    else:
                        # L'image est déjà au bon format
                        skipped_count += 1
            except Exception as e:
                # Afficher une erreur si une image est corrompue ou ne peut être traitée
                print(f"\nERREUR: Impossible de traiter le fichier '{image_path}'. Erreur: {e}")

        print(f"  -> Traitement terminé.")
        print(f"  -> {converted_count} images ont été converties en RGB.")
        print(f"  -> {skipped_count} images étaient déjà au bon format.")

if __name__ == "__main__":
    convert_grayscale_to_rgb(DATASET_ROOTS)
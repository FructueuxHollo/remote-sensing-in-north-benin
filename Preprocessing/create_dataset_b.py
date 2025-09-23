import os
import pandas as pd
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm
import random
import re
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---

# Dossier racine contenant tous les datasets tuilés
TILED_DATASETS_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Tiled")
# Dossier contenant les datasets SAVI tuilés et les métadonnées CSV
SAVI_DATASETS_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Tiled")

# Dossier où seront stockés les datasets B finaux
FINAL_DATASET_B_ROOT = Path(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Final")

# Configurations à traiter
CONFIGURATIONS = [
    {"size": "640x640", "savi_train_dir": "SAVI_train_tiled_640x640", "savi_test_dir": "SAVI_test_tiled_640x640"},
    {"size": "1024x1024", "savi_train_dir": "SAVI_train_tiled_1024x1024", "savi_test_dir": "SAVI_test_tiled_1024x1024"},
]

# --- FONCTIONS UTILITAIRES ---

def parse_hit_uav_filename(filename_stem):
    """Extrait les métadonnées du nom de fichier original de HIT-UAV."""
    parts = filename_stem.split('_')
    if len(parts) < 4: return None
    
    time_code = parts[0]
    altitude = parts[1]
    angle = parts[2]
    
    # Gérer les cas où T=0 (jour)
    meteo = "Sunny" if time_code == '0' else "Night"
    
    # Calcul de l'angle comme demandé
    try:
        # Angle = 90 - angle de la caméra
        final_angle = 90 - int(angle)
        return {"angle": final_angle, "altitude": int(altitude), "meteo": meteo}
    except ValueError:
        return None

def process_and_copy_files(file_list, source_prefix, dest_img_dir, dest_lbl_dir, all_metadata_rows, savi_metadata_df=None):
    """Copie les fichiers, les renomme et génère/récupère les métadonnées."""
    for source_img_path in tqdm(file_list, desc=f"Processing {source_prefix}"):
        original_stem = source_img_path.stem
        new_stem = f"{source_prefix}_{original_stem}"
        
        # Copier l'image et le label
        source_lbl_path = source_img_path.parent.parent.parent / "labels" / source_img_path.parent.name / f"{original_stem}.txt"
        dest_img_path = dest_img_dir / f"{new_stem}.jpg"
        dest_lbl_path = dest_lbl_dir / f"{new_stem}.txt"
        
        shutil.copy2(source_img_path, dest_img_path)
        if source_lbl_path.exists():
            shutil.copy2(source_lbl_path, dest_lbl_path)
        else: # Créer un fichier label vide si aucun n'existe
            dest_lbl_path.touch()

        # Générer/Récupérer les métadonnées
        row = {'id': new_stem}
        if source_prefix == "SAVI":
            meta = savi_metadata_df[savi_metadata_df['id'] == original_stem]
            if not meta.empty:
                row.update(meta.iloc[0].to_dict())
        else:
            if source_prefix == "HIT-UAV":
                parsed_meta = parse_hit_uav_filename(original_stem)
                if parsed_meta:
                    row.update(parsed_meta)
                row.update({'region': 'urban', 'mode': 'semi-automatique', 'y_start': 0, 'y_end': 512})
            elif source_prefix == "POP":
                with Image.open(source_img_path) as img:
                    _, img_h = img.size
                row.update({'angle': 0, 'altitude': 50, 'meteo': 'cloudy', 'region': 'urban periphery', 'mode': 'semi-automatique', 'y_start': 0, 'y_end': img_h})
        
        all_metadata_rows.append(row)

def process_and_copy_savi_files(file_list, source_prefix, dest_img_dir, dest_lbl_dir, all_metadata_rows, savi_metadata_df=None):
    """Copie les fichiers, les renomme et génère/récupère les métadonnées."""
    for source_img_path in tqdm(file_list, desc=f"Processing {source_prefix}"):
        original_stem = source_img_path.stem
        new_stem = original_stem
        
        # Copier l'image et le label
        source_lbl_path = source_img_path.parent / "labels" / f"{original_stem}.txt"
        dest_img_path = dest_img_dir / f"{new_stem}.jpg"
        dest_lbl_path = dest_lbl_dir / f"{new_stem}.txt"
        
        shutil.copy2(source_img_path, dest_img_path)
        if source_lbl_path.exists():
            shutil.copy2(source_lbl_path, dest_lbl_path)
        else: # Créer un fichier label vide si aucun n'existe
            dest_lbl_path.touch()

        # Générer/Récupérer les métadonnées
        row = {'id': new_stem}
        if source_prefix == "SAVI":
            meta = savi_metadata_df[savi_metadata_df['id'] == original_stem]
            if not meta.empty:
                row.update(meta.iloc[0].to_dict())
        else:
            if source_prefix == "HIT-UAV":
                parsed_meta = parse_hit_uav_filename(original_stem)
                if parsed_meta:
                    row.update(parsed_meta)
                row.update({'region': 'urban', 'mode': 'semi-automatique', 'y_start': 0, 'y_end': 512})
            elif source_prefix == "POP":
                with Image.open(source_img_path) as img:
                    _, img_h = img.size
                row.update({'angle': 0, 'altitude': 50, 'meteo': 'cloudy', 'region': 'urban periphery', 'mode': 'semi-automatique', 'y_start': 0, 'y_end': img_h})
        
        all_metadata_rows.append(row)

# --- SCRIPT PRINCIPAL ---

def create_final_dataset_b():
    for config in CONFIGURATIONS:
        size = config["size"]
        print(f"\n{'='*20} DÉBUT DE LA CRÉATION DU DATASET B - {size} {'='*20}")
        
        # 1. Préparer les chemins
        final_output_dir = FINAL_DATASET_B_ROOT / f"Dataset_B_{size}"
        # Créer les dossiers de destination finaux
        for split in ["train", "val", "test"]:
            (final_output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (final_output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # 2. Gérer SAVI et calculer les quotas
        savi_train_img_dir = SAVI_DATASETS_ROOT / config["savi_train_dir"] / "images"
        savi_test_img_dir = SAVI_DATASETS_ROOT / config["savi_test_dir"] / "images"
        
        # Charger les métadonnées de SAVI
        savi_train_metadata_df = pd.read_csv(SAVI_DATASETS_ROOT / f"savi_train_metadata_{size}.csv")
        savi_test_metadata_df = pd.read_csv(SAVI_DATASETS_ROOT / f"savi_test_metadata_{size}.csv") 

        savi_all_train_files = list(savi_train_img_dir.glob("*.jpg"))
        savi_test_files = list(savi_test_img_dir.glob("*.jpg"))
        
        # Répartir SAVI train en train/val
        savi_train_files, savi_val_files = train_test_split(savi_all_train_files, test_size=0.1, random_state=42)
        
        # Calculer les quotas pour HIT-UAV et POP
        quota_train = len(savi_train_files) // 2
        quota_val = len(savi_val_files) // 2
        quota_test = len(savi_test_files) // 2
        
        print(f"--- Quotas calculés pour les autres datasets ({size}) ---")
        print(f"  -> Train: {quota_train} images chacun")
        print(f"  -> Val:   {quota_val} images chacun")
        print(f"  -> Test:  {quota_test} images chacun")

        # 3. Échantillonner les fichiers de HIT-UAV et POP
        hit_uav_root = TILED_DATASETS_ROOT / f"HIT-UAV_tiled_{size}"
        pop_root = TILED_DATASETS_ROOT / f"POP_tiled_{size}"
        
        hit_uav_train_files = random.sample(list((hit_uav_root / "images" / "train").glob("*.jpg")), quota_train)
        hit_uav_val_files = random.sample(list((hit_uav_root / "images" / "val").glob("*.jpg")), quota_val)
        hit_uav_test_files = random.sample(list((hit_uav_root / "images" / "test").glob("*.jpg")), quota_test)

        pop_train_files = random.sample(list((pop_root / "images" / "train").glob("*.jpg")), quota_train)
        pop_val_files = random.sample(list((pop_root / "images" / "val").glob("*.jpg")), quota_val)
        pop_test_files = random.sample(list((pop_root / "images" / "test").glob("*.jpg")), quota_test)

        # 4. Traiter et assembler chaque split (train, val, test)
        all_metadata = []
        
        # TRAIN SET
        print("\n--- Assemblage du TRAIN set ---")
        dest_train_img = final_output_dir / "images" / "train"
        dest_train_lbl = final_output_dir / "labels" / "train"
        process_and_copy_savi_files(savi_train_files, "SAVI", dest_train_img, dest_train_lbl, all_metadata, savi_train_metadata_df)
        process_and_copy_files(hit_uav_train_files, "HIT-UAV", dest_train_img, dest_train_lbl, all_metadata)
        process_and_copy_files(pop_train_files, "POP", dest_train_img, dest_train_lbl, all_metadata)

        # VALIDATION SET
        print("\n--- Assemblage du VAL set ---")
        dest_val_img = final_output_dir / "images" / "val"
        dest_val_lbl = final_output_dir / "labels" / "val"
        process_and_copy_savi_files(savi_val_files, "SAVI", dest_val_img, dest_val_lbl, all_metadata, savi_train_metadata_df)
        process_and_copy_files(hit_uav_val_files, "HIT-UAV", dest_val_img, dest_val_lbl, all_metadata)
        process_and_copy_files(pop_val_files, "POP", dest_val_img, dest_val_lbl, all_metadata)
        
        # TEST SET
        print("\n--- Assemblage du TEST set ---")
        dest_test_img = final_output_dir / "images" / "test"
        dest_test_lbl = final_output_dir / "labels" / "test"
        process_and_copy_savi_files(savi_test_files, "SAVI", dest_test_img, dest_test_lbl, all_metadata, savi_test_metadata_df)
        process_and_copy_files(hit_uav_test_files, "HIT-UAV", dest_test_img, dest_test_lbl, all_metadata)
        process_and_copy_files(pop_test_files, "POP", dest_test_img, dest_test_lbl, all_metadata)
        
        # 5. Créer le CSV final des métadonnées
        final_metadata_df = pd.DataFrame(all_metadata)
        final_csv_path = final_output_dir / f"metadata_{size}.csv"
        final_metadata_df.to_csv(final_csv_path, index=False)
        print(f"\nDataset B pour la taille {size} créé avec succès.")
        print(f"Fichier de métadonnées final : {final_csv_path}")

if __name__ == "__main__":
    FINAL_DATASET_B_ROOT.mkdir(exist_ok=True)
    create_final_dataset_b()
    print(f"\n{'='*20} TOUS LES DATASETS B ONT ÉTÉ CRÉÉS {'='*20}")
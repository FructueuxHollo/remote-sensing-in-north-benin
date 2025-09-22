import re
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp', '.heic', '.heif'}

def rename_images(folder_path):
    """
    Renomme les images dans folder_path en ne gardant que les 4 chiffres centraux.
    Ex: dji_fly_20250819_090618_0001_1755591371578_photo.jpg -> 0001.jpg

    Param:
        folder_path (str or Path): chemin vers le dossier (non récursif)

    Retour:
        dict: résumé avec keys: renamed, skipped_not_image, skipped_no_match, errors (int)
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Le chemin spécifié n'existe pas ou n'est pas un dossier : {folder}")

    def is_image_file(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in IMAGE_EXTS

    def extract_four_digits(filename: str):
        # Cherche d'abord _####_ puis fallback sur _#### (avant extension ou fin)
        m = re.search(r'_(\d{4})_', filename)
        if m:
            return m.group(1)
        m2 = re.search(r'_(\d{4})(?:_|$)', filename)
        if m2:
            return m2.group(1)
        return None

    def unique_destination(dest: Path) -> Path:
        """
        Si dest existe, ajoute suffixe _1, _2, ... pour obtenir un nom unique.
        """
        if not dest.exists():
            return dest
        stem = dest.stem
        suffix = dest.suffix
        i = 1
        while True:
            candidate = dest.with_name(f"{stem}_{i}{suffix}")
            if not candidate.exists():
                return candidate
            i += 1

    renamed = 0
    skipped_not_image = 0
    skipped_no_match = 0
    errors = 0

    for p in folder.iterdir():
        try:
            if not p.is_file():
                continue
            if not is_image_file(p):
                skipped_not_image += 1
                continue

            code = extract_four_digits(p.name)
            if not code:
                skipped_no_match += 1
                continue

            new_name = f"{code}{p.suffix.lower()}"
            dest = p.with_name(new_name)

            # Si déjà le même fichier (nom identique et même chemin), on saute
            try:
                if p.resolve() == dest.resolve():
                    continue
            except Exception:
                # en cas de problème avec resolve(), on continue normalement

                pass

            dest = unique_destination(dest)

            p.rename(dest)
            print(f"RENAMED: {p.name} -> {dest.name}")
            renamed += 1

        except Exception as e:
            print(f"ERROR processing {p}: {e}")
            errors += 1

    summary = {
        "renamed": renamed,
        "skipped_not_image": skipped_not_image,
        "skipped_no_match": skipped_no_match,
        "errors": errors
    }
    print("-- Résumé --")
    print(summary)
    return summary
if __name__ == "__main__":
    rename_images(r"D:\Fructueux\Work\Memoire\Computer Vision\Material\Dataset\Originals\SAVI\Annotations\Tankpe_01\labels\train")
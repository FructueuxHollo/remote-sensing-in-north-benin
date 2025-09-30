#!/usr/bin/env python3
"""
save_rgb_channels.py
Lit une image, sépare les canaux R, G, B et enregistre :
 - une image couleur isolant chaque canal (ex: canal R en rouge)
 - optionnellement une image en niveaux de gris pour chaque canal

Usage:
    python save_rgb_channels.py path/to/image.jpg [--out-dir out] [--format png] [--grayscale]

Dépendances:
    pip install pillow numpy
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def save_channel_images(image_path, out_dir=".", fmt="png", save_grayscale=False, prefix=None):
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ouvrir et forcer en RGB (ignore alpha si présent)
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)  # shape (H, W, 3), dtype uint8

    base = prefix if prefix else image_path.stem

    # Préparer copies pour images couleur isolée
    r_col = arr.copy()
    r_col[..., 1] = 0
    r_col[..., 2] = 0

    g_col = arr.copy()
    g_col[..., 0] = 0
    g_col[..., 2] = 0

    b_col = arr.copy()
    b_col[..., 0] = 0
    b_col[..., 1] = 0

    # Convertir en PIL et sauvegarder (couleur)
    r_img = Image.fromarray(r_col)
    g_img = Image.fromarray(g_col)
    b_img = Image.fromarray(b_col)

    r_path = out_dir / f"{base}_R.{fmt}"
    g_path = out_dir / f"{base}_G.{fmt}"
    b_path = out_dir / f"{base}_B.{fmt}"

    r_img.save(r_path)
    g_img.save(g_path)
    b_img.save(b_path)

    saved = [str(r_path), str(g_path), str(b_path)]

    # Optionnel : sauvegarder les canaux en niveaux de gris (intensité)
    if save_grayscale:
        r_gray = Image.fromarray(arr[..., 0], mode="L")
        g_gray = Image.fromarray(arr[..., 1], mode="L")
        b_gray = Image.fromarray(arr[..., 2], mode="L")

        r_gray_path = out_dir / f"{base}_R_gray.{fmt}"
        g_gray_path = out_dir / f"{base}_G_gray.{fmt}"
        b_gray_path = out_dir / f"{base}_B_gray.{fmt}"

        r_gray.save(r_gray_path)
        g_gray.save(g_gray_path)
        b_gray.save(b_gray_path)

        saved += [str(r_gray_path), str(g_gray_path), str(b_gray_path)]

    return saved


def main():
    saved = save_channel_images(r"D:\Fructueux\Work\Projet\vortexcrypt\test\test_images\bart_test_image.png", out_dir=r"D:\Téléchargements")

    print("Fichiers enregistrés :")
    for s in saved:
        print(" -", s)


if __name__ == "__main__":
    main()

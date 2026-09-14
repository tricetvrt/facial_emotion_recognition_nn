"""
undersample.py — nasumicno PREMESTA slike iz trening skupa za klase koje
imaju vise od TARGET_COUNT uzoraka (po argmax hard labeli) u podfolder
"visak" unutar istog IMAGE_DIR, tako da nijedna klasa ne prelazi
TARGET_COUNT. Klase ispod TARGET_COUNT se ne diraju.

CSV se ne dira - postojeci FERPlusDataset loader vec preskace redove ciji
fajl ne postoji na originalnoj putanji ("fajl ne postoji na disku"), pa
premestene slike automatski ispadaju iz treninga bez ikakve izmene CSV-a.
Reverzibilno je - samo vrati fajlove iz "visak" nazad ako zatreba.
"""

import os
import shutil
import random

from fer_csv_dataset import FERPlusDataset, KEPT_CLASSES
import config

TARGET_COUNT = 5000
IMAGE_DIR = "../dataAugmented/Clean/"         
VISAK_DIR = os.path.join(IMAGE_DIR, "visak")


def main():
    random.seed(config.SEED)
    os.makedirs(VISAK_DIR, exist_ok=True)

    print("Ucitavanje train skupa (ista filtracija kao pri treningu)...")
    train_ds = FERPlusDataset(
        csv_path=config.TRAIN_CSV,
        image_dir=IMAGE_DIR,
        usage="Training",
        transform=None,
    )

    by_class = {c: [] for c in KEPT_CLASSES}
    for (path, _soft_label), hard in zip(train_ds.samples, train_ds.hard_targets):
        by_class[KEPT_CLASSES[hard]].append(path)

    total_moved = 0

    for class_name in KEPT_CLASSES:
        paths = by_class[class_name]
        n = len(paths)

        if n <= TARGET_COUNT:
            print(f"{class_name:10s}: {n} <= {TARGET_COUNT} -> ne dira se")
            continue

        n_to_move = n - TARGET_COUNT
        chosen = random.sample(paths, n_to_move)
        print(f"{class_name:10s}: {n} -> premestam {n_to_move} u 'visak', ostaje {TARGET_COUNT}")

        for path in chosen:
            dst = os.path.join(VISAK_DIR, os.path.basename(path))
            shutil.move(path, dst)
            total_moved += 1

    print(f"\nUkupno premesteno u '{VISAK_DIR}': {total_moved} slika")


if __name__ == "__main__":
    main()
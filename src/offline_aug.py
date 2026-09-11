import os
import csv
from PIL import Image
from torchvision import transforms
import shutil

from fer_csv_dataset import FERPlusDataset, KEPT_CLASSES, FERPLUS_COLUMNS
import config

TARGET_COUNT = 7000
OUTPUT_CSV = "../dataAugmented/fer2013new_augmented.csv"

offline_aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=127),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
])


def load_raw_votes_by_image_name(csv_path):
    """image_name -> lista od 10 sirovih glasova (string), tacno kako stoji u originalnom CSV-u."""
    votes_by_name = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 2 + len(FERPLUS_COLUMNS):
                continue
            image_name = row[1].strip()
            if image_name:
                votes_by_name[image_name] = row[2:2 + len(FERPLUS_COLUMNS)]
    return votes_by_name


def main():
    os.makedirs(config.TRAIN_AUG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print("Ucitavanje train skupa (ista filtracija kao pri treningu)...")
    train_ds = FERPlusDataset(
        csv_path=config.TRAIN_CSV,
        image_dir=config.TRAIN_IMAGE_DIR,
        usage="Training",
        transform=None,
    )
    raw_votes_by_name = load_raw_votes_by_image_name(config.TRAIN_CSV)

    by_class = {c: [] for c in KEPT_CLASSES}
    for (path, _soft_label), hard in zip(train_ds.samples, train_ds.hard_targets):
        by_class[KEPT_CLASSES[hard]].append(path)

    new_rows = []
    total_generated = 0

    for class_name in KEPT_CLASSES:
        originals = by_class[class_name]
        n_original = len(originals)

        if n_original >= TARGET_COUNT:
            print(f"{class_name:10s}: {n_original} >= {TARGET_COUNT} -> preskace se")
            continue

        n_needed = TARGET_COUNT - n_original
        print(f"{class_name:10s}: {n_original} originala -> generisem {n_needed} augmentovanih")

        for i in range(n_needed):
            src_path = originals[i % n_original]
            src_name = os.path.basename(src_path)
            src_votes = raw_votes_by_name[src_name]   # kopirani glasovi sa originalne slike

            image = Image.open(src_path).convert("RGB")
            aug_image = offline_aug_transform(image)

            base_name = os.path.splitext(src_name)[0]
            new_name = f"{base_name}_aug{i}.png"
            aug_image.save(os.path.join(config.TRAIN_AUG_DIR, new_name))

            new_rows.append(["Training", new_name] + src_votes)
            total_generated += 1

    print(f"\nUkupno generisano: {total_generated} novih slika")

    with open(config.TRAIN_CSV, newline="", encoding="utf-8") as f_in:
        all_rows = list(csv.reader(f_in))
    all_rows.extend(new_rows)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        csv.writer(f_out).writerows(all_rows)

    print(f"Nov CSV sacuvan: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

#redosled
FERPLUS_COLUMNS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt", "unknown", "NF",
]

# zadrzavamo ove klase samo
KEPT_CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "fear"]

DISPLAY_NAME = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "surprise",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
}


class FERPlusSoftDataset(Dataset):
    def __init__(self, csv_path, image_dir, usage, transform=None, min_votes=1):
        self.image_dir = image_dir
        self.transform = transform
        self.classes = KEPT_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(KEPT_CLASSES)}

        self.samples = []      # lista (putanja_do_slike, soft_label_np_array)
        self.hard_targets = [] # za WeightedRandomSampler / izvestaje (argmax klase)

        kept_col_idx = [FERPLUS_COLUMNS.index(c) for c in KEPT_CLASSES]

        skipped_no_image = 0
        skipped_not_face = 0
        skipped_zero_votes = 0
        skipped_missing_file = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2 + len(FERPLUS_COLUMNS):
                    continue
                row_usage, image_name = row[0].strip(), row[1].strip()
                if row_usage != usage:
                    continue
                if not image_name:
                    skipped_no_image += 1
                    continue

                votes = [int(v) for v in row[2:2 + len(FERPLUS_COLUMNS)]]
                unknown_votes = votes[FERPLUS_COLUMNS.index("unknown")]
                nf_votes = votes[FERPLUS_COLUMNS.index("NF")]

                #ako je NF/unknown vecinski glas nije upotrebljiva slika
                if max(unknown_votes, nf_votes) >= max(votes):
                    skipped_not_face += 1
                    continue

                kept_votes = np.array([votes[i] for i in kept_col_idx], dtype=np.float32)
                total = kept_votes.sum()
                if total < min_votes:
                    skipped_zero_votes += 1
                    continue

                soft_label = kept_votes / total  # normalizacija na sumu 1

                image_path = os.path.join(image_dir, image_name)
                if not os.path.isfile(image_path):
                    skipped_missing_file += 1
                    continue

                self.samples.append((image_path, soft_label))
                self.hard_targets.append(int(np.argmax(soft_label)))

        print(f"[{usage}] Ucitano: {len(self.samples)} slika")
        print(f"  Preskoceno - nema imena slike: {skipped_no_image}")
        print(f"  Preskoceno - NF/unknown vecinski: {skipped_not_face}")
        print(f"  Preskoceno - 0 glasova posle filtriranja: {skipped_zero_votes}")
        print(f"  Preskoceno - fajl ne postoji na disku: {skipped_missing_file}")

        counts = np.bincount(self.hard_targets, minlength=len(KEPT_CLASSES))
        print("  Raspodela (po argmax hard labeli, samo informativno):")
        for c, n in zip(KEPT_CLASSES, counts):
            print(f"    {DISPLAY_NAME[c]:10s}: {n}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, soft_label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(soft_label, dtype=torch.float32)
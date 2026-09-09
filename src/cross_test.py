

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE, MODEL_SAVE_PATH, NUM_CLASSES
from model import get_resnet50
from transforms import val_transform  # NIKAD augmentacija za evaluaciju
from dataset import get_class_names   # ["neutral","happiness","surprise","sadness","anger","fear"]

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
KEPT_CLASSES = get_class_names()
CLASS_TO_IDX = {c: i for i, c in enumerate(KEPT_CLASSES)}



AFFECTNET_MAPPING = {
    "neutral": "neutral",
    "happy": "happiness",
    "sad": "sadness",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": None,     # izbacujemo, kao i kod FER+
    "anger": "anger",
    "contempt": None,    # izbacujemo, kao i kod FER+
}

# RAF-DB standardno koristi brojne foldere 1-7 (vidi header liste iznad)
RAFDB_MAPPING = {
    "1": "surprise",
    "2": "fear",
    "3": None,        # disgust - izbacujemo
    "4": "happiness",
    "5": "sadness",
    "6": "anger",
    "7": "neutral",
}


class RemappedImageFolder(Dataset):
    """
    Wrapper oko torchvision ImageFolder koji:
      1. Mapira originalne foldere/klase na nasih 6 klasa preko label_mapping
      2. PRESKACE slike cija klasa se mapira na None (npr. disgust/contempt)
    """
    def __init__(self, root, label_mapping, transform=None):
        base_dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.samples = []

        skipped = 0
        for path, folder_idx in base_dataset.samples:
            folder_name = base_dataset.classes[folder_idx]
            mapped_name = label_mapping.get(folder_name)

            if mapped_name is None:
                skipped += 1
                continue

            target_idx = CLASS_TO_IDX[mapped_name]
            self.samples.append((path, target_idx))

        print(f"  Ucitano: {len(self.samples)} slika, preskoceno (nemapirane klase): {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_model(path):
    model = get_resnet50(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader):
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def evaluate_dataset(model, dataset_name, root, label_mapping, batch_size=64, num_workers=4):
    print(f"\n{'='*60}\nCross-test: {dataset_name}\n{'='*60}")
    print(f"Ucitavanje sa: {root}")

    dataset = RemappedImageFolder(root=root, label_mapping=label_mapping, transform=val_transform)
    if len(dataset) == 0:
        print(f"UPOZORENJE: 0 slika ucitano za {dataset_name} - proveri putanju/mapiranje.")
        return None

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    y_true, y_pred = predict(model, loader)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nMacro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nclassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(KEPT_CLASSES))),
        target_names=KEPT_CLASSES,
        zero_division=0,
    ))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(KEPT_CLASSES))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # izbegava deljenje sa 0

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=KEPT_CLASSES, yticklabels=KEPT_CLASSES)
    plt.title(f"Confusion Matrix (normalizovano) - {dataset_name}")
    plt.xlabel("predikcija")
    plt.ylabel("tacna klasa")
    plt.tight_layout()
    save_path = f"cross_test_{dataset_name.lower().replace('-', '')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Matrica sacuvana: {save_path}")

    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "n_samples": len(dataset)}


def main():
    print(f"Koriscen uredjaj: {device}")
    print(f"Ucitavanje modela: {MODEL_SAVE_PATH}")
    model = load_model(MODEL_SAVE_PATH)

    AFFECTNET_ROOT = "../data2/affectnetTEST" 
    RAFDB_ROOT = "../data2/rafdbTEST"

    results = {}

    if os.path.isdir(AFFECTNET_ROOT):
        results["AffectNet"] = evaluate_dataset(
            model, "AffectNet", AFFECTNET_ROOT, AFFECTNET_MAPPING
        )
    else:
        print(f"\nPreskoceno AffectNet - putanja ne postoji: {AFFECTNET_ROOT}")

    if os.path.isdir(RAFDB_ROOT):
        results["RAF-DB"] = evaluate_dataset(
            model, "RAF-DB", RAFDB_ROOT, RAFDB_MAPPING
        )
    else:
        print(f"\nPreskoceno RAF-DB - putanja ne postoji: {RAFDB_ROOT}")

    print(f"\n{'='*60}\nREZIME CROSS-DATASET TESTIRANJA\n{'='*60}")
    print(f"{'Dataset':<15} {'N slika':<10} {'Macro F1':<12} {'Weighted F1':<12}")
    for name, r in results.items():
        if r:
            print(f"{name:<15} {r['n_samples']:<10} {r['macro_f1']:<12.4f} {r['weighted_f1']:<12.4f}")


if __name__ == "__main__":
    main()
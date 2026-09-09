import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV,          # putanje do fer2013new.csv (ili isti fajl, filtrirano po Usage koloni)
    TRAIN_IMAGE_DIR, VAL_IMAGE_DIR, TEST_IMAGE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)
from transforms import train_transform, val_transform
from fer_csv_dataset import FERPlusSoftDataset, KEPT_CLASSES


def get_train_dataset():
    return FERPlusSoftDataset(
        csv_path=TRAIN_CSV,
        image_dir=TRAIN_IMAGE_DIR,
        usage="Training",
        transform=train_transform,
    )


def get_validation_dataset():
    return FERPlusSoftDataset(
        csv_path=VAL_CSV,
        image_dir=VAL_IMAGE_DIR,
        usage="PublicTest",
        transform=val_transform,
    )


def get_test_dataset():
    return FERPlusSoftDataset(
        csv_path=TEST_CSV,
        image_dir=TEST_IMAGE_DIR,
        usage="PrivateTest",
        transform=val_transform,
    )


def make_weighted_sampler(dataset, power=0.5):
    """Balansira frekvenciju na osnovu argmax (hard) labele svake slike."""
    targets = np.array(dataset.hard_targets)
    class_counts = np.bincount(targets, minlength=len(KEPT_CLASSES))
    class_weights = 1.0 / (class_counts ** power)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(use_weighted_sampler=True, sampler_power=0.5):
    train_dataset = get_train_dataset()
    val_dataset = get_validation_dataset()
    test_dataset = get_test_dataset()
    pin_memory = torch.cuda.is_available()

    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset, power=sampler_power)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,          # sampler i shuffle se ne kombinuju
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def get_class_names():
    return KEPT_CLASSES


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Classes:", get_class_names())

    images, soft_labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)
    print("Batch soft-label shape:", soft_labels.shape)
    print("Primer soft labele:", soft_labels[0])
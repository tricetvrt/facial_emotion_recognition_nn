import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from dataset import get_dataloaders
from model import get_resnet50, get_mobilenetv4
from tqdm import tqdm


def soft_ce_loss(logits, target_distribution):
    """
    Soft cross-entropy: -sum(target * log_softmax(logits))
    target_distribution je vektor glasova normalizovan na sumu 1 (soft label),
    NE one-hot vektor.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_distribution * log_probs).sum(dim=1).mean()


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d) and "classifier" not in name:
            module.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, soft_labels in progress_bar:
        images = images.to(device)
        soft_labels = soft_labels.to(device)          # shape [B, num_classes], suma=1 po redu

        optimizer.zero_grad()
        outputs = model(images)
        loss = soft_ce_loss(outputs, soft_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # za "accuracy" tokom treninga poredimo argmax predikcije sa argmax soft labele
        # (samo informativno - stvarna metrika je macro F1 na kraju, ne accuracy)
        predicted = outputs.argmax(dim=1)
        target_hard = soft_labels.argmax(dim=1)
        total += images.size(0)
        correct += predicted.eq(target_hard).sum().item()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })
    return running_loss / total, correct / total


def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, soft_labels in progress_bar:
            images = images.to(device)
            soft_labels = soft_labels.to(device)

            outputs = model(images)
            loss = soft_ce_loss(outputs, soft_labels)
            running_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(dim=1)
            target_hard = soft_labels.argmax(dim=1)
            total += images.size(0)
            correct += predicted.eq(target_hard).sum().item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })
    return running_loss / total, correct / total


def main():
    torch.manual_seed(config.SEED)
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Korišćeni uređaj za trening: {device}")

    print("Učitavanje data loader-a (CSV soft labele + weighted sampler)...")
    train_loader, val_loader, _ = get_dataloaders(
        use_weighted_sampler=True,
        sampler_power=0.5,
    )

    print(f"Inicijalizacija modela {config.MODEL_NAME} za Feature Extraction...")
    model = get_mobilenetv4(
        num_classes=config.NUM_CLASSES,      #6 klasa
        pretrained=config.PRETRAINED,
        freeze_backbone=config.FREEZE_BACKBONE,
        model_name=config.MODEL_NAME         
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trenabilno: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # EARLY STOPPING PODESAVANJA
    MAX_EPOCHS = 50          # gornja granica, cak i ako model i dalje napreduje
    PATIENCE = 7             # koliko epoha zaredom tolerisemo bez ubedljivog poboljsanja
    MIN_DELTA = 0.001        # koliko val_loss MORA da se smanji da bi se racunalo kao poboljsanje
                              

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    print(f"\n--- Pocetak treninga (max {MAX_EPOCHS} epoha, patience={PATIENCE}, min_delta={MIN_DELTA}) ---")
    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)

        print(
            f"Epoha [{epoch + 1}/{MAX_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        # da li je poboljsanje ubedljivo ne samo sitan sum
        if val_loss < best_val_loss - MIN_DELTA:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  -> Novi najbolji model (val_loss poboljsan za {improvement:.4f}). Sacuvan u: {config.MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"  -> Nema ubedljivog poboljsanja ({epochs_no_improve}/{PATIENCE} epoha zaredom)")

        if epochs_no_improve >= PATIENCE:
            print(
                f"\nEarly stopping: nema ubedljivog poboljsanja val_loss-a "
                f"u poslednjih {PATIENCE} epoha (best_val_loss={best_val_loss:.4f})."
            )
            break

    print("\nTrening zavrsen!")


if __name__ == "__main__":
    main()
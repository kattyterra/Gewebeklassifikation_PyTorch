import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast, GradScaler
import pandas as pd


def parse_args():
    """
    Liest und parst die Kommandozeilenargumente für das Training.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, required=True, help="Path to training images (subfolders = classes)")
    p.add_argument("--val_dir", type=str, required=True, help="Path to validation images (subfolders = classes)")
    p.add_argument("--img", type=int, default=384, help="Image size. Default 384")
    p.add_argument("--batch", type=int, default=16, help="Batch size. Default 16")
    p.add_argument("--epochs", type=int, default=10, help="Total epochs (across both phases). Default 10")
    p.add_argument("--freeze_epochs", type=int, default=5, help="Epochs with backbone frozen. Default 5")
    p.add_argument("--finetune_layers", type=int, default=60, help="How many top layers to unfreeze in phase 2. Default 60")
    p.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for phase 1")
    p.add_argument("--lr_ft", type=float, default=1e-5, help="Learning rate for phase 2")
    p.add_argument("--no_class_weights", action="store_true", help="Disable automatic class weights")
    p.add_argument("--outdir", type=str, default="./runs_v2s_torch", help="Output directory")
    return p.parse_args()


def count_images_by_class(root):
    """
    Zählt Bilder pro Klasse in einem Datenordner.
    """
    counts = {}
    for cls in sorted(os.listdir(root)):
        dpath = os.path.join(root, cls)
        if os.path.isdir(dpath):
            counts[cls] = len([
                f for f in os.listdir(dpath)
                if f.lower().endswith((".png", ".tif"))
            ])
    print(counts)
    print("Total:", sum(counts.values()))
    return counts


def compute_class_weights(train_dir, class_names):
    """
    Berechnet Klassen-Gewichte basierend auf Häufigkeiten im Trainingsdatensatz.
    """
    counts = count_images_by_class(train_dir)
    total = sum(counts.get(c, 0) for c in class_names)
    weights = []
    for c in class_names:
        n = counts.get(c, 1)  # Fallback 1
        weights.append(total / (len(class_names) * n))
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(train_dir, val_dir, img, batch, class_weights=None, num_workers=4, use_cuda=True):
    """
    Erstellt DataLoader für Training und Validierung.
    """
    # Offizielle Preprocessing-Transforms der ImageNet-Weights
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    base_transforms = weights.transforms()

    # Wir hängen nur Resize + Flip davor
    transform_train = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.RandomHorizontalFlip(),
        base_transforms,  # enthält ToTensor + Normalize
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img, img)),
        base_transforms,
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transform_val)

    if class_weights is not None:
        targets = [y for _, y in train_ds.samples]
        weights_samples = [class_weights[t] for t in targets]
        sampler = WeightedRandomSampler(weights_samples, num_samples=len(weights_samples), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    return train_loader, val_loader, train_ds.classes


def make_model(num_classes):
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    model = models.efficientnet_b3(weights=weights)
    # Kopf anpassen
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    """
    Führt genau eine Trainings-Epoche aus (mit optionaler Mixed Precision).
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device, use_amp):
    """
    Evaluiert das Modell auf einem gegebenen DataLoader.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return running_loss / total, correct / total, y_true, y_pred


def main():
    """
    Haupt-Trainingsroutine mit CUDA-Optimierung & AMP.
    """
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_amp = use_cuda  # AMP nur auf CUDA sinnvoll

    print("Device:", device)
    if use_cuda:
        torch.backends.cudnn.benchmark = True  # schnellere Convs bei festen Inputgrößen

    # Klassen-Gewichte berechnen, wenn nicht explizit deaktiviert
    class_weights = None
    if not args.no_class_weights:
        tmp_ds = datasets.ImageFolder(args.train_dir)
        class_weights = compute_class_weights(args.train_dir, tmp_ds.classes)

    # DataLoader und Klassenliste erzeugen
    train_loader, val_loader, class_names = build_dataloaders(
        args.train_dir,
        args.val_dir,
        args.img,
        args.batch,
        class_weights=class_weights,
        num_workers=4,
        use_cuda=use_cuda,
    )

    # Modell & Mixed-Precision-Scaler
    model = make_model(len(class_names)).to(device)
    scaler = GradScaler(enabled=use_amp)

    # Phase 1 – Feature-Backbone einfrieren (nur Klassifikationskopf trainieren)
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    history = []

    # ----------------- Phase 1: Frozen Backbone -----------------
    for epoch in range(args.freeze_epochs):
        print(f"\nEpoch {epoch+1}/{args.freeze_epochs} (Frozen)")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device, use_amp)

        scheduler.step(val_loss)
        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

        pd.DataFrame(history, columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]).to_csv(
            os.path.join(args.outdir, "history_phase1.csv"), index=False
        )

    # Zwischenstand des Modells nach Phase 1 sichern
    torch.save(model.state_dict(), os.path.join(args.outdir, "best_phase1.pt"))

    # Phase 2 – Fine-Tuning: oberste finetune_layers Layer des Feature-Backbones freigeben
    for param in model.features[-args.finetune_layers:].parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=args.lr_ft)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    # ----------------- Phase 2: Fine-Tuning -----------------
    for epoch in range(args.freeze_epochs, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} (Fine-tuning)")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device, use_amp)

        scheduler.step(val_loss)
        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

        pd.DataFrame(history, columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]).to_csv(
            os.path.join(args.outdir, "history_phase2.csv"), index=False
        )

    # Confusion Matrix (absolut und normalisiert) berechnen und speichern
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(args.outdir, "confusion_matrix.csv")
    )
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        os.path.join(args.outdir, "confusion_matrix_normalized.csv")
    )

    # Klassennamen separat speichern
    with open(os.path.join(args.outdir, "class_names.txt"), "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print("class_names.txt gespeichert:", os.path.join(args.outdir, "class_names.txt"))

    # Finales Modell sichern
    torch.save(model.state_dict(), os.path.join(args.outdir, "final_model.pt"))
    print("Training complete. Results saved in:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()

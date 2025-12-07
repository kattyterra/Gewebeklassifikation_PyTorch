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
from torch import amp
import pandas as pd


def parse_args():
    """
    Liest und parst die Kommandozeilenargumente für das Training.

    Argumente:
    --train_dir        : Pfad zu Trainingsbildern (Unterordner = Klassen)
    --val_dir          : Pfad zu Validierungsbildern (Unterordner = Klassen)
    --img              : Bildgröße (Quadrat, z.B. 384x384)
    --batch            : Batchgröße
    --epochs           : Gesamtanzahl der Epochen (Phase 1 + Phase 2)
    --freeze_epochs    : Anzahl Epochen in Phase 1 (Feature-Backbone eingefroren)
    --finetune_layers  : Anzahl "oberer" Layer, die in Phase 2 freigegeben werden
    --lr_head          : Lernrate für Phase 1 (nur Klassifikationskopf)
    --lr_ft            : Lernrate für Phase 2 (Fine-Tuning)
    --no_class_weights : Wenn gesetzt, keine automatischen Klassen-Gewichte
    --outdir           : Ausgabeverzeichnis für Logs, Confusion Matrix, Modelle etc.
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

    Erwartet Ordnerstruktur:
        root/
          KlasseA/
          KlasseB/
          ...

    Es werden nur Dateien mit Endung .png oder .tif gezählt.
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

    Idee: inverse Häufigkeit:
      weight_c = total_samples / (num_classes * count_c)

    Wird zur Gewichtung der Cross-Entropy-Loss verwendet, um Klassenimbalance zu kompensieren.
    """
    counts = count_images_by_class(train_dir)
    total = sum(counts.get(c, 0) for c in class_names)
    weights = []
    for c in class_names:
        n = counts.get(c, 1)  # Fallback 1, falls Klasse nicht im Dict
        weights.append(total / (len(class_names) * n))
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(train_dir, val_dir, img, batch, class_weights=None):
    """
    Erstellt DataLoader für Training und Validierung.

    - Wendet einfache Augmentierung nur im Training an (Horizontal Flip).
    - Wenn class_weights gesetzt sind:
        → WeightedRandomSampler wird verwendet, um Klassenimbalance auszugleichen.
    """
    transform_train = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transform_val)

    if class_weights is not None:
        # Sample-Gewichte basierend auf Klassen-Gewichten
        targets = [y for _, y in train_ds.samples]
        weights = [class_weights[t] for t in targets]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=4)
    return train_loader, val_loader, train_ds.classes


def make_model(num_classes):
    """
    Erstellt ein EfficientNetV2-S Modell mit ImageNet-Vortrainierung
    und ersetzt die Klassifikationsschicht durch eine neue mit num_classes Ausgängen.
    """
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# Train / Eval mit Mixed Precision
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """
    Führt genau eine Trainings-Epoche aus.

    - Mixed Precision (torch.amp) für schnellere und speichereffizientere Berechnung
    - Gibt durchschnittlichen Loss und Accuracy über die Epoche zurück
    """
    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixed precision forward/backward
        with amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        # Skaliertes Backward für Mixed Precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """
    Evaluiert das Modell auf einem gegebenen DataLoader.

    Gibt zurück:
    - durchschnittlichen Validierungs-Loss
    - Accuracy
    - y_true : Liste wahrer Klassenlabels
    - y_pred : Liste vorhergesagter Klassenlabels
    (für Confusion Matrix / weitere Auswertungen)
    """
    model.eval()
    running_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad(), amp.autocast('cuda'):
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
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
    Haupt-Trainingsroutine:

    - Argumente parsen
    - Klassengewichte (optional) berechnen
    - DataLoader bauen
    - Modell erstellen
    - 2-Phasen-Training:
        Phase 1: Backbone eingefroren, nur Kopf trainieren
        Phase 2: oberste Layer des Backbones finetunen
    - Trainingshistorie, Confusion Matrix, Klassennamen und Modelle speichern
    """
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Klassen-Gewichte berechnen, wenn nicht explizit deaktiviert
    class_weights = None
    if not args.no_class_weights:
        tmp_model = datasets.ImageFolder(args.train_dir)
        class_weights = compute_class_weights(args.train_dir, tmp_model.classes)

    # DataLoader und Klassenliste erzeugen
    train_loader, val_loader, class_names = build_dataloaders(
        args.train_dir, args.val_dir, args.img, args.batch, class_weights
    )

    # Modell & Mixed-Precision-Scaler
    model = make_model(len(class_names)).to(device)
    scaler = amp.GradScaler('cuda')

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
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

        # Trainingshistorie der ersten Phase fortlaufend speichern
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
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

        # Trainingshistorie der zweiten Phase fortlaufend speichern
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

    # Klassennamen separat speichern (für spätere Inferenz-Skripte)
    with open(os.path.join(args.outdir, "class_names.txt"), "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print("class_names.txt gespeichert:", os.path.join(args.outdir, "class_names.txt"))

    # Finales Modell sichern
    torch.save(model.state_dict(), os.path.join(args.outdir, "final_model.pt"))
    print("Training complete. Results saved in:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()

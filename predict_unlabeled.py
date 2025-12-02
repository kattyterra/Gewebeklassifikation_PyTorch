import os
import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from torch import nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Pfad zum trainierten Modell (.pt)")
    p.add_argument("--img_path", type=str, required=True, help="Pfad zu einem Bild oder Ordner mit Bildern")
    p.add_argument("--img_size", type=int, default=384, help="Bildgröße, wie im Training")
    return p.parse_args()


def load_model(model_path, num_classes, device):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model


def predict_image(model, img_path, transform, class_names, device):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0] 
        pred_idx = probs.argmax()
        pred_class = class_names[pred_idx]
        return pred_class, probs


def main():
    args = parse_args()

    out_csv = args.img_path + "/" + os.path.basename(args.img_path) + "_unformatted_results.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" Device:", device)

    # Klassennamen: wie im Training (WICHTIG: Neue Klassen hier hinzufügen!!!)
    class_names = ["Epithel", "Snp", "Stroma", "xtra"]

    # Modell laden
    model = load_model(args.model_path, len(class_names), device)

    # Transformation wie im Training
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # Bilddateien sammeln (auch rekursiv)
    image_files = []
    if os.path.isdir(args.img_path):
        for root, _, files in os.walk(args.img_path):
            for f in files:
                if f.lower().endswith((".png", ".tif", ".tiff")):
                    image_files.append(os.path.join(root, f))
    else:
        image_files = [args.img_path]

    if not image_files:
        print("Keine Bilddateien gefunden.")
        return

    results = []
    print(f"\n {len(image_files)} Bilder gefunden. Starte Klassifizierung...\n")

    for img_path in image_files:
        pred_class, probs = predict_image(model, img_path, transform, class_names, device)
        result = {
            "filename": os.path.basename(img_path),
            "predicted_class": pred_class,
        }
        # jede Klassenwahrscheinlichkeit hinzufügen
        for cls_name, p in zip(class_names, probs):
            result[f"prob_{cls_name}"] = round(p * 100, 2)
        results.append(result)


    # CSV speichern
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\n Ergebnisse gespeichert in: {os.path.abspath(out_csv)}")

    # Eingabedatei lesen
    df = pd.read_csv(out_csv)

    # Mapping von Klassennamen zu Zahlen
    class_map = {name: idx for idx, name in enumerate(class_names)}

    # Fehlende Spalten prüfen
    required_cols = ["prob_Epithel", "prob_Snp", "prob_Stroma", "prob_xtra", "predicted_class"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' fehlt in der Eingabedatei {out_csv}")

    # Neue DataFrame-Struktur
    new_df = pd.DataFrame({
        "prob_Epithel": df["prob_Epithel"].astype(float),
        "prob_Snp": df["prob_Snp"].astype(float),
        "prob_Stroma": df["prob_Stroma"].astype(float),
        "prob_xtra": df["prob_xtra"].astype(float),
        "predicted_class_num": df["predicted_class"].map(class_map)
    })

    formatted_path = args.img_path + "/" + os.path.basename(args.img_path) + "_results.csv"

    # Ausgabeordner erstellen
    os.makedirs(os.path.dirname(formatted_path) or ".", exist_ok=True)

    # Neue CSV ohne Kopfzeile speichern
    new_df.to_csv(formatted_path, index=False, sep=";", header=False)

    print(f"Neue CSV gespeichert in: {os.path.abspath(formatted_path)}")


if __name__ == "__main__":
    main()

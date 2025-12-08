import os
import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from torch import nn


def parse_args():
    """
    Liest Kommandozeilenargumente ein.

    --model_path : Pfad zur trainierten Modell-Datei (.pt)
    --img_path   : Pfad zu einem Bild oder Ordner mit Bildern
    --img_size   : Ziel-Bildgröße wie im Training
    """

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Pfad zum trainierten Modell (.pt)")
    p.add_argument("--img_path", type=str, required=True, help="Pfad zu einem Bild oder Ordner mit Bildern")
    p.add_argument("--img_size", type=int, default=384, help="Bildgröße, wie im Training")
    return p.parse_args()


def load_model(model_path, num_classes, device):
    """
    Lädt ein EfficientNetV2-S Modell, ersetzt die Klassifikationsschicht
    und lädt die trainierten Gewichte.

    model_path : Pfad zur .pt-Datei
    num_classes : Anzahl der Klassenausgänge
    device : CPU oder GPU
    """

    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model


def predict_image(model, img_path, transform, class_names, device):
    """
    Führt eine Vorhersage für ein einzelnes Bild durch und liefert:

    - predicted_class : Name der vorhergesagten Klasse
    - probs           : Softmax-Wahrscheinlichkeitsvektor
    """

    img = Image.open(img_path).convert("RGB")

    # Bild vorbereiten
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)

        # Softmax-Wahrscheinlichkeiten berechnen
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        pred_class = class_names[pred_idx]
        return pred_class, probs


def main():
    """ Hauptpipeline: Modell laden, Bilder klassifizieren, CSV erzeugen. """

    args = parse_args()

    # Standardausgabe-CSV
    out_csv = args.img_path + "/" + os.path.basename(args.img_path) + "_unformatted_results.csv"

    # Gerät wählen (GPU falls verfügbar)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" Device:", device)

    # Klassennamen wie im Training (falls neue Klassen existieren → hier erweitern)
    class_names = ["Epithel", "Snp", "Stroma", "xtra"]

    # Modell laden
    model = load_model(args.model_path, len(class_names), device)

    # Transformationen wie beim Training
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # Bilddateien sammeln (rekursiv durchs Ordner)
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

    print(f"\n {len(image_files)} Bilder gefunden. Starte Klassifizierung...\n")

    results = []

    # Hauptschleife: Alle Bilder klassifizieren
    for img_path in image_files:
        pred_class, probs = predict_image(model, img_path, transform, class_names, device)

        # Rohdatensatz vorbereiten
        result = {
            "filename": os.path.basename(img_path),
            "predicted_class": pred_class,
        }

        # Alle Klassenwahrscheinlichkeiten ergänzen
        for cls_name, p in zip(class_names, probs):
            result[f"prob_{cls_name}"] = round(p * 100, 2)

        results.append(result)

    # Zwischenergebnis speichern
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\n Ergebnisse gespeichert in: {os.path.abspath(out_csv)}")

    # Eingabe-CSV lesen
    df = pd.read_csv(out_csv)

    # Mapping: Klassennamen → numerische IDs
    class_map = {name: idx for idx, name in enumerate(class_names)}

    # Prüfen, ob alle relevanten Spalten existieren
    required_cols = ["prob_Epithel", "prob_Snp", "prob_Stroma", "prob_xtra", "predicted_class"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' fehlt in der Eingabedatei {out_csv}")

    # Formatierte Zieldatei erzeugen
    new_df = pd.DataFrame({
        "prob_Epithel": df["prob_Epithel"].astype(float),
        "prob_Snp": df["prob_Snp"].astype(float),
        "prob_Stroma": df["prob_Stroma"].astype(float),
        "prob_xtra": df["prob_xtra"].astype(float),
        "predicted_class_num": df["predicted_class"].map(class_map)
    })

    formatted_path = args.img_path + "/" + os.path.basename(args.img_path) + "_results.csv"

    # Neue CSV ohne Kopfzeile speichern
    new_df.to_csv(formatted_path, index=False, sep=";", header=False)
    print(f"Neue CSV gespeichert in: {os.path.abspath(formatted_path)}")


if __name__ == "__main__":
    main()

import os
import argparse
import torch
from torchvision import transforms, models
from torchvision.models import EfficientNet_B3_Weights
from PIL import Image
import pandas as pd
from torch import nn


def parse_args():
    """
    Liest die Kommandozeilenargumente ein.
        -f : Pfad zu einer Bilddatei oder einem Ordner mit Bildern
        -img_size : Bildgröße für das Modell (muss dem Training entsprechen)
    """
    p = argparse.ArgumentParser()
    p.add_argument("-f", type=str, required=True, help="Pfad zu einem Bild oder Ordner mit Bildern")
    p.add_argument("-img_size", type=int, default=384, help="Bildgröße, wie im Training")
    return p.parse_args()


def load_model(model_path, num_classes, device):
    """
    Lädt ein EfficientNet-B3 Modell, passt die Klassifikationsschicht an
    und lädt die gespeicherten Gewichte.
        model_path : Pfad zur .pt-Modelldatei
        num_classes : Anzahl der Klassenausgänge
        device : CPU oder GPU
    """
    # Kein Pretrained-Weight, da wir eigene Gewichte laden
    model = models.efficientnet_b3(weights=None)

    # Klassifikator anpassen → Anzahl Klassen ändern (wie im Training)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Modellparameter laden
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()            # Inference-Modus
    model.to(device)        # Modell auf das Zielgerät verschieben
    return model


def predict_image(model, img_path, transform, class_names, device):
    """
    Nimmt ein einzelnes Bild, wandelt es in ein Tensorformat um,
    führt eine Vorhersage durch und gibt die Klasse + Wahrscheinlichkeiten zurück.
    """
    # Bild laden und RGB erzwingen
    img = Image.open(img_path).convert("RGB")

    # Transformation anwenden und Batch-Dimension hinzufügen
    img_t = transform(img).unsqueeze(0).to(device)

    # Vorhersage ohne Gradientenberechnung
    with torch.no_grad():
        outputs = model(img_t)

        # Softmax-Wahrscheinlichkeiten berechnen
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        pred_class = class_names[pred_idx]
        return pred_class, probs


def main():
    """ Hauptfunktion: Modell laden, Bilder einlesen, Vorhersagen erzeugen und CSV speichern. """

    args = parse_args()

    # Gerät auswählen (GPU bevorzugt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Klassennamen exakt wie im Training
    class_names = ["Epithel", "Snp", "Stroma", "xtra"]

    # Modellpfad relativ zum Skript
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "final_model.pt")

    # Modell laden
    model = load_model(model_path, len(class_names), device)

    # Gleiche Preprocessing-Transforms wie beim Training mit EfficientNet-B3
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    base_transforms = weights.transforms()

    # Für Inferenz: Resize + offizielle Normalisierung, aber keine Augmentation
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        base_transforms,  # enthält ToTensor + Normalize
    ])

    # Bilddateien auflisten
    image_files = []
    if os.path.isdir(args.f):
        # Falls ein Ordner angegeben wurde → rekursiv durchsuchen
        for root, _, files in os.walk(args.f):
            for f in files:
                if f.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, f))
    else:
        # Einzelbild
        image_files = [args.f]

    if not image_files:
        print("Keine gültigen Bilddateien gefunden.")
        return

    results = []

    # Für jedes Bild eine Modellvorhersage durchführen
    for img_path in image_files:
        pred_class, probs = predict_image(model, img_path, transform, class_names, device)

        # Numerische Klassen-ID bestimmen
        class_idx = class_names.index(pred_class)

        # CSV-Zeile erzeugen
        row = {
            "prob_Epithel": round(float(probs[class_names.index("Epithel")]), 2),
            "prob_Snp":     round(float(probs[class_names.index("Snp")]), 2),
            "prob_Stroma":  round(float(probs[class_names.index("Stroma")]), 2),
            "prob_xtra":    round(float(probs[class_names.index("xtra")]), 2),
            "predicted_class_num": class_idx
        }

        results.append(row)

    # DataFrame für die Ausgabe erzeugen
    new_df = pd.DataFrame(results)

    # Ausgabepfad konstruieren
    parent_folder = os.path.abspath(os.path.join(args.f, ".."))
    folder_name = os.path.basename(args.f.rstrip(os.sep))
    formatted_path = os.path.join(parent_folder, folder_name + "_results.csv")

    # CSV speichern (ohne Kopfzeile, wie in deinem Original)
    new_df.to_csv(formatted_path, index=False, sep=";", header=False)
    print("Ergebnisse gespeichert unter:", formatted_path)


if __name__ == "__main__":
    main()

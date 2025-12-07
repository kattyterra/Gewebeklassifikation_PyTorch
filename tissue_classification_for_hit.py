import os
import argparse
import torch
from torchvision import transforms, models
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
    Lädt ein EfficientNetV2-S Modell, passt die Klassifikationsschicht an
    und lädt die gespeicherten Gewichte.
        model_path : Pfad zur .pt-Modelldatei  
        num_classes : Anzahl der Klassenausgänge  
        device : CPU oder GPU
    """

    model = models.efficientnet_v2_s(weights=None)

    # Klassifikator anpassen → Anzahl Klassen ändern
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Modellparameter laden
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()          # Inference-Modus
    model.to(device)      # Modell auf das Zielgerät verschieben
    return model


def predict_image(model, img_path, transform, class_names, device):
    """
    Nimmt ein einzelnes Bild, wandelt es in ein Tensorformat um,
    führt eine Vorhersage durch und gibt die Klasse + Wahrscheinlichkeiten zurück.
        img_path : Bildpfad  
        transform : angewandte Bildtransformationen  
        class_names : Liste der Klassennamen  
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

    # Klassennamen exakt wie im Training (WICHTIG: bei neuen Klassen anpassen!)
    class_names = ["Epithel", "Snp", "Stroma", "xtra"]

    # Modellpfad relativ zum Skript
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "final_model.pt")

    # Modell laden
    model = load_model(model_path, len(class_names), device)

    # Transformationen wie im Training
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # Bilddateien auflisten
    image_files = []
    if os.path.isdir(args.f):
        # Falls ein Ordner angegeben wurde → rekursiv durchsuchen
        for root, _, files in os.walk(args.f):
            for f in files:
                if f.lower().endswith((".png", ".tif", ".tiff")):
                    image_files.append(os.path.join(root, f))
    else:
        # Einzelbild
        image_files = [args.f]

    if not image_files:
        return  # Keine gültigen Dateien vorhanden

    results = []

    # Für jedes Bild eine Modellvorhersage durchführen
    for img_path in image_files:
        pred_class, probs = predict_image(model, img_path, transform, class_names, device)

        # Numerische Klassen-ID bestimmen
        class_idx = class_names.index(pred_class)

        #CSV-Zeile erzeugen
        row = {
            "prob_Epithel": round(probs[class_names.index("Epithel")] * 100, 2),
            "prob_Snp": round(probs[class_names.index("Snp")] * 100, 2),
            "prob_Stroma": round(probs[class_names.index("Stroma")] * 100, 2),
            "prob_xtra": round(probs[class_names.index("xtra")] * 100, 2),
            "predicted_class_num": class_idx
        }

        results.append(row)

    # DataFrame für die Ausgabe erzeugen
    new_df = pd.DataFrame(results)

    # Ausgabepfad konstruieren
    parent_folder = os.path.abspath(os.path.join(args.f, ".."))
    folder_name = os.path.basename(args.f)
    formatted_path = os.path.join(parent_folder, folder_name + "_results.csv")

    # CSV speichern (ohne Kopfzeile)
    new_df.to_csv(formatted_path, index=False, sep=";", header=False)


if __name__ == "__main__":
    main()

import os
import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from torch import nn      


def parse_args():
    """
    Liest alle Kommandozeilenargumente ein.

    --model_path   : Pfad zur trainierten .pt-Datei  
    --class_names  : Textdatei mit Klassennamen (eine Klasse pro Zeile)  
    --img_path     : Pfad zu einem Bild oder Ordner mit Bildern  
                     (Unterordnername = echte Klasse, z.B. dataset/Epithel/img1.png)
    --img_size     : Ziel-Bildgröße wie im Training  
    --out_csv      : Ausgabe-CSV mit Vorhersagen  
    """

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Pfad zum trainierten Modell (.pt)")
    p.add_argument("--class_names", type=str, required=True, help="Pfad zu einer Textdatei mit Klassennamen (eine pro Zeile)")
    p.add_argument("--img_path", type=str, required=True, help="Pfad zu einem Bild oder Ordner mit Bildern (Unterordner = echte Klassen)")
    p.add_argument("--img_size", type=int, default=384, help="Bildgröße, wie im Training")
    p.add_argument("--out_csv", type=str, default="predictions/preds.csv", help="Pfad zur Ausgabedatei (CSV)")
    return p.parse_args()


def load_model(model_path, num_classes, device):
    """
    Lädt ein EfficientNetV2-S Modell und ersetzt die Klassifikationsschicht
    durch eine neue Schicht mit der gewünschten Anzahl von Klassen.

    model_path : Pfad zur .pt-Datei  
    num_classes : Anzahl der Ausgangsklassen  
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
    Führt eine Vorhersage für ein einzelnes Bild durch.

    Rückgabe:
    - predicted_class : Klassenname mit höchster Wahrscheinlichkeit
    - confidence      : numerischer Softmax-Wert der höchsten Wahrscheinlichkeit
    """

    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        pred_class = class_names[pred.item()]
        return pred_class, conf.item()


def main():
    """
    - Klassen laden
    - Modell laden
    - Bilder sammeln
    - Vorhersagen durchführen
    - CSV speichern & Genauigkeit berechnen
    """
    
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Klassennamen aus Datei lesen
    with open(args.class_names, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    # Modell laden
    model = load_model(args.model_path, len(class_names), device)

    # Transformationen wie im Training
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # Alle Bilddateien rekursiv sammeln
    image_files = []
    
    # Ordner → über Unterordner iterieren (Unterordner = echte Klasse)
    for root, _, files in os.walk(args.img_path):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root, f))

    results = []

    # Vorhersage für jedes Bild
    for img_path in image_files:

        # Echte Klasse = Name des übergeordneten Ordners
        true_class = os.path.basename(os.path.dirname(img_path))

        pred_class, conf = predict_image(model, img_path, transform, class_names, device)

        results.append({
            "filename": os.path.basename(img_path),
            "true_class": true_class,
            "predicted_class": pred_class,
            "confidence": round(conf * 100, 2),
            "correct": pred_class == true_class
        })

    # Ergebnisse als CSV speichern
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)

    # Genauigkeit berechnen
    acc = df["correct"].mean() * 100
    print(f"\n Ergebnisse gespeichert in: {os.path.abspath(args.out_csv)}")
    print(f"Gesamtgenauigkeit: {acc:.2f}%")


if __name__ == "__main__":
    main()

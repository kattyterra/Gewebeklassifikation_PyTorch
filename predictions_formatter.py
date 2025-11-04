import argparse
import pandas as pd
import os

def parse_args():
    p = argparse.ArgumentParser(description="Erstellt neue CSV ohne Kopfzeile mit Wahrscheinlichkeiten und Klassenindex")
    p.add_argument("--input_csv", type=str, required=True, help="Pfad zur Eingabe-CSV (z. B. predictions_unlabeled.csv)")
    p.add_argument("--output_csv", type=str, default="formatted_predictions.csv", help="Pfad zur Ausgabe-CSV")
    return p.parse_args()


def main():
    args = parse_args()

    # Eingabedatei lesen
    df = pd.read_csv(args.input_csv)

    # Mapping von Klassennamen zu Zahlen
    class_map = {
        "Epithel": 0,
        "Snp": 1,
        "Stroma": 2
    }

    # Fehlende Spalten prüfen
    required_cols = ["prob_Epithel", "prob_Snp", "prob_Stroma", "predicted_class"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' fehlt in der Eingabedatei {args.input_csv}")

    # Neue DataFrame-Struktur
    new_df = pd.DataFrame({
        "prob_Epithel": df["prob_Epithel"],
        "prob_Snp": df["prob_Snp"],
        "prob_Stroma": df["prob_Stroma"],
        "predicted_class_num": df["predicted_class"].map(class_map)
    })

    # Ausgabeordner erstellen
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Neue CSV ohne Kopfzeile speichern
    new_df.to_csv(args.output_csv, index=False, header=False)

    print(f"Neue CSV gespeichert in: {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()

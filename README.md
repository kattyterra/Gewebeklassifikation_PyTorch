# Gewebeklassifikationstool auf Basis von EfficientNetV2-S

Dieses Projekt enthält ein vollständiges Deep-Learning-System zur Klassifikation von den CLSM-Aufnahmen.
Es umfasst ein zweiphasiges Trainingsskript (Backbone eingefroren, danach Fine-Tuning), mehrere Inferenzskripte sowie CSV-Ausgaben für nachgelagerte Verarbeitungsschritte. Mixed Precision Training und optionales Class Weighting werden unterstützt.


## Projektübersicht

- project/

    - bild_daten/ - Sammlung an Datensätzen für Training und Evaluation
    
    - ergebnisse/ - Ordnersammlung mit Ergebnissdateien aus dem Training

    - train.py Trainingsskript mit zwei Phasen

    - predict.py - Klassifiziert Aufnahmen in Unterordner sortiert (Unterordner -> echte Klasse)

    - predict_unlabeled.py - Klassifiziert unsortierte Aufnahmen

    - tissue_classification_for_hit.py - Unsortierte Klassifizierung für HIT angepasst

    - final_model.pt - Das im Moment verwendetes Modell zur Bildklassifikation (WICHTIG für die HIT-Klassifikation)

    - README.md


## Installation

Python Version 3.9 oder höher wird empfohlen.

Falls PyTorch noch nicht installiert ist:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


## Training starten

Das Training besteht aus zwei Phasen:

- Backbone eingefroren, nur Klassifikationskopf wird trainiert.

- Fine-Tuning der oberen Layer.

Beispielaufruf:
python train.py --train_dir bild_daten/training --val_dir bild_daten/evaluation --epochs 20 --freeze_epochs 10 --outdir ergebnisse/test1

Das Skript erzeugt folgende Dateien:
- history_phase1.csv
- history_phase2.csv
- confusion_matrix.csv
- confusion_matrix_normalized.csv
- class_names.txt
- best_phase1.pt
- final_model.pt


## Klassifikation für ein einzelnes Bild

python infer_simple.py --model_path final_model.pt --img_path example.png --img_size 384

## Klassifikation für Ordner

Das Skript erkennt die echte Klasse automatisch anhand des Ordnernamens.

python infer_folder.py --model_path final_model.pt --class_names class_names.txt --img_path dataset/test

Beispielausgabedatei preds.csv enthält:
filename, true_class, predicted_class, confidence, correct

## Formatierte CSV-Ausgaben

infer_formatted.py erzeugt zwei Dateien:
_unformatted_results.csv Rohwerte und Wahrscheinlichkeiten
_results.csv Formatiert ohne Header, für KI-Pipelines geeignet

Beispielaufruf:
python infer_formatted.py --model_path final_model.pt --img_path data/test

Format der formatierten CSV:
prob_Epithel ; prob_Snp ; prob_Stroma ; prob_xtra ; predicted_class_num

## Datensatzstruktur

Die Daten müssen wie folgt organisiert sein:

data/
train/
Epithel/
Snp/
Stroma/
xtra/
val/
Epithel/
Snp/
Stroma/
xtra/

Klassennamen-Datei

Nach dem Training wird eine Datei class_names.txt erzeugt, mit folgendem Format:

Epithel
Snp
Stroma
xtra

Ergebnisse und Ausgaben

Trainingsverläufe (Loss und Accuracy)

Absolute und normalisierte Confusion Matrix

Finales trainiertes Modell

Inferenz-CSV-Dateien

Optional: Klassen-Gewichtung und Weighted Sampling
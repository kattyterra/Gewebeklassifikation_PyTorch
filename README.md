# Gewebeklassifikationstool auf Basis von EfficientNetV2-S

Dieses Projekt enthält ein vollständiges Deep-Learning-System zur Klassifikation von CLSM-Aufnahmen.  
Es umfasst ein zweiphasiges Trainingsskript (Backbone eingefroren, anschließend Fine-Tuning), mehrere Inferenzskripte sowie CSV-Ausgaben für nachgelagerte Verarbeitungsschritte. Mixed-Precision-Training und optionales Class-Weighting werden unterstützt.

## Projektübersicht

- **project/**
    - **bild_daten/** – Sammlung der Datensätze für Training und Evaluation
    - **ergebnisse/** – Ordner mit Ergebnisdateien aus dem Training
    - **train.py** – Trainingsskript mit zwei Phasen
    - **predict.py** – Klassifiziert Aufnahmen, die in Unterordnern nach echter Klasse sortiert sind
    - **predict_unlabeled.py** – Klassifiziert unsortierte Aufnahmen
    - **tissue_classification_for_hit.py** – Unsortierte Klassifikation, angepasst für HIT
    - **final_model.pt** – Aktuell verwendetes Modell zur Bildklassifikation (wichtig für die HIT-Klassifikation)
    - **README.md**

## Installation

Python Version 3.9 oder höher wird empfohlen.

Falls PyTorch noch nicht installiert ist:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Training starten

Das Training besteht aus zwei Phasen:

- Backbone eingefroren, nur Klassifikationskopf wird trainiert.

- Fine-Tuning der oberen Layer.

**Beispielaufruf**:
```bash
python train.py --train_dir bild_daten/training --val_dir bild_daten/evaluation --epochs 20 --freeze_epochs 10 --outdir ergebnisse/test1
```

Das Skript erzeugt folgende Dateien:

- **history_phase1.csv** -> Trainingsverlauf Phase 1 (Loss und Accuracy)
- **history_phase2.csv** -> Trainingsverlauf Phase 2 (Loss und Accuracy)
- **confusion_matrix.csv** -> Absolute Confusion-Matrix
- **confusion_matrix_normalized.csv** -> Normalisierte Confusion-Matrix
- **class_names.txt** -> Textdatei mit Klassennamen
- **best_phase1.pt** -> Bestes Modell aus Phase 1
- **final_model.pt** -> Finales Modell

## Datensatzstruktur für das Training

Die Daten für das Training müssen wie folgt organisiert sein:

- evaluation/
    - /Epithel/
        - img1.png (auch .tif möglich)
        - img2.png 
        - ...
    - /Snp/
        - -//-
    - /Stroma/
        - -//-
    - /xtra/
        - -//-

- training/
    - /Epithel/
        - -//-
    - /Snp/
        - -//-
    - /Stroma/
        - -//-
    - /xtra/
        - -//-



## Klassifikation für ein einzelnes Bild

```bash
python predict_unlabeled.py --model_path final_model.pt --img_path example.png --img_size 384
```

oder

```bash
python tissue_classification_for_hit.py -f example.png
```

## Klassifikation eines Ordners (Aufnahmen nach Klassen vorsortiert)

Das Skript **predict.py** erkennt die echte Klasse automatisch anhand des Ordnernamens ( => Datensatzstruktur wie im Training! ) :

```bash
python predict.py --model_path final_model.pt --class_names class_names.txt --img_path dataset/test --out_csv predictions/preds.csv
```

Beispielausgabedatei **preds.csv** enthält:
filename, true_class, predicted_class, confidence, correct

## Klassifikation eines Ordners (Aufnahmen nicht vorsoritert)

Das Skript **predict_unlabeled.py** erzeugt zwei Dateien:

- **_unformatted_results.csv** -> Rohwerte und Wahrscheinlichkeiten (filename, predicted_class, prob_Epithel, prob_Snp, prob_Stroma, prob_xtra, predicted_class (Klassenname als String))

- **_results.csv** -> Formatiert ohne Header, für die Weiterverwendung in HIT geeignet (alle Wahrscheinlichkeitswerte als Float mit 2 Nachkommastellen, als Prozentangabe; predicted_class als Integer gemäß der Trainingsreihenfolge)

**Beispielaufruf:**
```bash
python infer_formatted.py --model_path final_model.pt --img_path data/test
```

**Integer-Werte für Klassen:**
 - Epithel = 0
 - Snp = 1
 - Stroma = 2
 - xtra = 3


## Gewebeklassifikation für HIT

Die Datei **tissue_classification_for_hit.py** wurde für die Verwendung in der HIT-Software angepasst.
Wenn in der HIT-Parameterdatenbank der Pfad zu XPIWIT durch den Pfad zu dieser Datei ersetzt wird und der Aufruf in
**HRTImagingToolDialog::ClassifyDataset()** wie folgt angepasst wird: 

```c++
auto result = CUtilities::Exec(L"python \"" + xpiwit + L"\" -f \"" + imageFolder.wstring() + L"\""); 
```

, wird HIT es zur Gewebeklassifikation verwenden.

**Wichtig:**
Das Modell zur Klassifikation muss **final_model.pt** heißen und sich im selben Ordner wie das Skript **tissue_classification_for_hit.py** befinden, damit es korrekt geladen werden kann.

Es wird eine formatierte **_results.csv** erzeugt (identisch mit der von predict_unlabeled.py). Diese Datei wird dann von HIT weiter verarbeitet und für die Mosaikerstellung verwendet.



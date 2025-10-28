# pyright: reportArgumentType=false

import os
import argparse
import numpy as np
import tensorflow as tf
import keras

from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.applications import EfficientNetV2S
from keras.optimizers import Adam

from typing import cast

# Optional: mixed precision if available
try:
    from keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, required=True, help="Path to training images (subfolders = classes)")
    p.add_argument("--val_dir", type=str, required=True, help="Path to validation images (subfolders = classes)")
    p.add_argument("--img", type=int, default=384, help="Image size (square). Default 384")
    p.add_argument("--batch", type=int, default=16, help="Batch size. Default 16")
    p.add_argument("--epochs", type=int, default=10, help="Total epochs (across both phases). Default 10")
    p.add_argument("--freeze_epochs", type=int, default=5, help="Epochs with backbone frozen. Default 5")
    p.add_argument("--finetune_layers", type=int, default=60, help="How many top layers to unfreeze in phase 2. Default 60")
    p.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for phase 1")
    p.add_argument("--lr_ft", type=float, default=1e-5, help="Learning rate for phase 2")
    p.add_argument("--no_class_weights", action="store_true", help="Disable automatic class weights")
    p.add_argument("--outdir", type=str, default="./runs_v2s", help="Output directory")
    return p.parse_args()


# Trainings- und Evaluationsdatensätze aus den Ordnern laden #
def build_datasets(train_dir, val_dir, img, batch):
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",  
        image_size=(img, img),
        batch_size=batch,
        shuffle=True
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(img, img),
        batch_size=batch,
        shuffle=False
    )

    train_ds = cast(tf.data.Dataset, train_ds)
    val_ds   = cast(tf.data.Dataset, val_ds)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# Ein Klassifikations-Modell auf Basis von EfficientNetV2S bauen #
def make_model(num_classes, img):
    base = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(img, img, 3),
        pooling="avg"
    )
    x = layers.Dropout(0.25)(base.output)
    dtype = "float32"
    out = layers.Dense(num_classes, activation="softmax", dtype=dtype)(x)
    model = models.Model(base.input, out)
    return model, base


# Zählt, wie viele Bilder in jeder Klasse es gibt #
def count_images_by_class(root):
    counts = {}
    for cls in sorted(os.listdir(root)):
        dpath = os.path.join(root, cls)
        if not os.path.isdir(dpath):
            continue

        c = sum(
            f.lower().endswith(".png")
            for _, _, files in os.walk(dpath)
            for f in files
        )
        counts[cls] = c

    print(counts)
    print("Total:", sum(counts.values()))

    return counts


# automatischen Klassengewichtung, damit ein unausgeglichenes Dataset fair behandelt wird #
def compute_class_weights(train_dir, class_names):
    counts = count_images_by_class(train_dir)
    if not counts:
        return None
    
    total = sum(counts.get(c, 0) for c in class_names)
    weights = {}
    for idx, c in enumerate(class_names):
        n = counts.get(c, 0)
        if n == 0:
            weights[idx] = 0.0
        else:
            weights[idx] = total / (len(class_names) * n)
    return weights



def train(args):
    os.makedirs(args.outdir, exist_ok=True)
    train_ds, val_ds, class_names = build_datasets(args.train_dir, args.val_dir, args.img, args.batch)
    num_classes = len(class_names)  

    # Class weights (optional)
    class_weight = None
    if not args.no_class_weights:
        class_weight = compute_class_weights(args.train_dir, class_names)

    model, base = make_model(num_classes, args.img)

    # Phase 1: freeze backbone
    for l in base.layers:
        l.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=args.lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt = ModelCheckpoint(os.path.join(args.outdir, "best_phase1.keras"), 
                           monitor="val_accuracy", 
                           mode="max", 
                           save_best_only=True, 
                           verbose=1)
    
    csvlog = CSVLogger(os.path.join(args.outdir, "history_phase1.csv"))
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)

    epochs_phase1 = min(args.freeze_epochs, args.epochs)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        class_weight=class_weight,
        callbacks=[ckpt, csvlog, es, rlrop],
        verbose=1
    )

    # Phase 2: fine-tune top layers
    for l in base.layers[-args.finetune_layers:]:
        l.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr_ft),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    ckpt2 = ModelCheckpoint(os.path.join(args.outdir, "best_phase2.keras"),
                            monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
    csvlog2 = CSVLogger(os.path.join(args.outdir, "history_phase2.csv"))
    es2 = EarlyStopping(monitor="val_loss", 
                        patience=8, 
                        restore_best_weights=True, 
                        verbose=1)
    
    rlrop2 = ReduceLROnPlateau(monitor="val_loss", 
                               factor=0.2, 
                               patience=3, 
                               verbose=1)

    remaining_epochs = max(args.epochs - epochs_phase1, 0)
    if remaining_epochs > 0:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining_epochs,
            class_weight=class_weight,
            callbacks=[ckpt2, csvlog2, es2, rlrop2],
            verbose=1
        )


    # Evaluate and save final model
    model.save(os.path.join(args.outdir, "final_model.keras"))
    val_metrics = model.evaluate(val_ds, verbose=0)
    print("Validation metrics [loss, accuracy]:", val_metrics)


    # Create confusion matrix on val
    y_true = []
    y_pred = []

    for x, y in val_ds: # pyright: ignore[reportGeneralTypeIssues]
        preds = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

    # Save raw confusion matrix
    cm_path = os.path.join(args.outdir, "confusion_matrix.csv")
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(class_names) + "\n")  # Header
        for i, cls in enumerate(class_names):
            f.write(cls + "," + ",".join(str(v) for v in cm[i]) + "\n")
    print("Saved:", cm_path)

    # Save normalized confusion matrix (percent per true class)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm_path = os.path.join(args.outdir, "confusion_matrix_normalized.csv")
    with open(cm_norm_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(class_names) + "\n")  # Header
        for i, cls in enumerate(class_names):
            f.write(cls + "," + ",".join(f"{v:.4f}" for v in cm_norm[i]) + "\n")
    print("Saved:", cm_norm_path)

    # Save label mapping
    with open(os.path.join(args.outdir, "class_names.txt"), "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print("Saved to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    args = parse_args()
    train(args)

import os
import argparse
from PIL import Image

def convert_tif_to_png(src_root, dst_root):
    """
    Konvertiert rekursiv alle .tif Bilder aus src_root zu .png
    und speichert sie in dst_root mit gleicher Ordnerstruktur.
    Bereits existierende .png Dateien werden nicht überschrieben.
    """
    for dirpath, _, files in os.walk(src_root):
        rel_path = os.path.relpath(dirpath, src_root)
        out_dir = os.path.join(dst_root, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        for file in files:
            if not file.lower().endswith(".tif"):
                continue

            src_file = os.path.join(dirpath, file)
            dst_file = os.path.join(out_dir, file.rsplit(".", 1)[0] + ".png")

            if os.path.exists(dst_file):
                print(f"[skip] {dst_file} bereits vorhanden")
                continue

            try:
                with Image.open(src_file) as img:
                    img = img.convert("RGB")
                    img.save(dst_file, "PNG", compress_level=3)
                    print(f"[ok]   {dst_file}")
            except Exception as e:
                print(f"[fail] {src_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Konvertiert TIFF zu PNG rekursiv.")
    parser.add_argument("--src", type=str, required=True, help="Quellordner mit .tif Bildern")
    parser.add_argument("--dst", type=str, required=True, help="Zielordner für .png Ausgabe")
    args = parser.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f"Quellordner nicht gefunden: {args.src}")

    os.makedirs(args.dst, exist_ok=True)

    convert_tif_to_png(args.src, args.dst)
    print("\n Process finished")
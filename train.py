"""
Fine-tuning de YOLOv8s sur un jeu de données de scarabées japonais.

Usage :
    .venv/bin/python train.py --data data/japanese_beetle.yaml --epochs 100

Le meilleur modèle est copié dans models/japanese_beetle.pt, que l'app charge automatiquement.
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/japanese_beetle.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="cuda, mps, cpu (auto par défaut)")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    # Si le projet a été déplacé, le chemin absolu du jeu dans le YAML est périmé : on le recale.
    data_yaml = Path(args.data).resolve()
    cfg = yaml.safe_load(data_yaml.read_text())
    ds = Path(cfg["path"])
    if not ds.is_absolute() or not ds.is_dir():
        local = data_yaml.parent / ds.name
        if local.is_dir():
            cfg["path"] = str(local)
            data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    args.data = str(data_yaml)

    model = YOLO("yolov8s.pt")  # poids COCO pré-entraînés comme point de départ
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=25,
        project=str(Path(__file__).parent.resolve() / "runs"),
        name="japanese_beetle",
        exist_ok=True,
        # Augmentations adaptées aux photos de feuillage prises du dessus
        flipud=0.5,
        fliplr=0.5,
        degrees=15,
        mosaic=1.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
    )

    # Chemin réel fourni par l'entraîneur (l'arborescence runs/ varie selon les versions d'Ultralytics).
    best = Path(model.trainer.best)
    if not best.is_file():
        raise SystemExit(f"best.pt introuvable ({best}).")
    dest = Path(__file__).parent.resolve() / "models" / "japanese_beetle.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    metrics = model.trainer.metrics or {}
    print(f"\nRésultats complets : {best.parent.parent}")
    print(f"mAP50 = {metrics.get('metrics/mAP50(B)', float('nan')):.3f} · "
          f"mAP50-95 = {metrics.get('metrics/mAP50-95(B)', float('nan')):.3f}")
    print(f"Modèle copié dans {dest} — redémarrez l'app.")


if __name__ == "__main__":
    main()

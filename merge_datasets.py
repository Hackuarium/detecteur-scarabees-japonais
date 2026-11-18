"""
Fusionne plusieurs exports Roboflow « YOLOv8 » en un seul jeu mono-classe japanese_beetle.

Usage :
    .venv/bin/python merge_datasets.py               # lit downloads/*.zip et downloads/*/
    .venv/bin/python merge_datasets.py a.zip b.zip   # ou des chemins explicites

Règles :
- Toute classe dont le nom contient « japanese » devient la classe 0 (japanese_beetle).
- Les annotations des autres insectes (coccinelle, punaise…) sont retirées, mais leurs
  images sont conservées : elles servent d'exemples négatifs (« ceci n'est pas un
  scarabée japonais »). Option --drop-others-images pour les exclure.
- Les découpages train/valid/test d'origine sont conservés (évite qu'une image et ses
  copies augmentées par Roboflow se retrouvent à la fois en train et en valid).
  Un export sans valid est découpé 85/15.
- Les images en double (même contenu) sont ignorées.
- Les annotations en polygone sont converties en boîtes.

Résultat : data/japanese_beetle/ + data/japanese_beetle.yaml (chemin absolu).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import yaml

BASE = Path(__file__).parent.resolve()
OUT = BASE / "data" / "japanese_beetle"
YAML_OUT = BASE / "data" / "japanese_beetle.yaml"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def find_root(d: Path) -> Path:
    """Dossier contenant data.yaml (un ZIP peut avoir un sous-dossier)."""
    hits = sorted(d.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not hits:
        raise SystemExit(f"Pas de data.yaml dans {d} — est-ce bien un export « YOLOv8 » ?")
    return hits[0].parent


def class_names(root: Path) -> dict[int, str]:
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def convert_label(text: str, mapping: dict[int, int]) -> list[str]:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in mapping:
            continue
        vals = [float(v) for v in parts[1:]]
        if len(vals) > 4:  # polygone x1 y1 x2 y2 … -> boîte englobante
            xs, ys = vals[0::2], vals[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            vals = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
        cx, cy, w, h = vals[:4]
        if w <= 0 or h <= 0:
            continue
        out.append(f"{mapping[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def collect(root: Path):
    """Liste (split, image, label|None) en conservant les splits d'origine."""
    items = []
    for sub in root.iterdir():
        split = SPLITS.get(sub.name.lower())
        if not split or not (sub / "images").is_dir():
            continue
        for img in sorted((sub / "images").iterdir()):
            if img.suffix.lower() in IMG_EXT:
                lab = sub / "labels" / (img.stem + ".txt")
                items.append([split, img, lab if lab.is_file() else None])
    if items and not any(s == "valid" for s, _, _ in items):
        rnd = random.Random(0)
        train = [it for it in items if it[0] == "train"]
        for it in rnd.sample(train, max(1, int(len(train) * 0.15))):
            it[0] = "valid"
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--drop-others-images", action="store_true",
                    help="exclure les images qui ne contiennent que d'autres insectes")
    ap.add_argument("--max-neg-ratio", type=float, default=0.2,
                    help="part maximale d'images sans scarabée dans train (défaut 0.2)")
    args = ap.parse_args()

    sources = [Path(s) for s in args.sources] or sorted(
        p for p in (BASE / "downloads").glob("*") if p.suffix == ".zip" or p.is_dir())
    if not sources:
        raise SystemExit("Aucun export trouvé. Déposez les ZIP « YOLOv8 » dans downloads/.")

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "valid", "test"):
        (OUT / s / "images").mkdir(parents=True)
        (OUT / s / "labels").mkdir(parents=True)

    seen: set[str] = set()
    stats = Counter()
    with tempfile.TemporaryDirectory() as tmp:
        for i, src in enumerate(sources):
            if src.suffix == ".zip":
                d = Path(tmp) / f"src{i}"
                with zipfile.ZipFile(src) as z:
                    z.extractall(d)
            else:
                d = src
            root = find_root(d)
            names = class_names(root)
            mapping = {k: 0 for k, n in names.items() if "japanese" in n.lower()}
            print(f"\n[{src.name}] classes : {list(names.values())}")
            if not mapping:
                print("  ⚠ aucune classe « japanese… » : toutes les images deviennent des négatifs")
            per = Counter()
            for split, img, lab in collect(root):
                h = hashlib.md5(img.read_bytes()).hexdigest()
                if h in seen:
                    per["doublons"] += 1
                    continue
                raw = lab.read_text() if lab else ""
                lines = convert_label(raw, mapping)
                if not lines and raw.strip() and args.drop_others_images:
                    per["ignorées (autres insectes)"] += 1
                    continue
                seen.add(h)
                name = f"d{i}_{img.stem}"
                shutil.copy(img, OUT / split / "images" / (name + img.suffix.lower()))
                (OUT / split / "labels" / (name + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
                per[f"{split}"] += 1
                per["boîtes"] += len(lines)
                per["négatives" if not lines else "positives"] += 1
            print("  " + " · ".join(f"{k}: {v}" for k, v in sorted(per.items())))
            stats.update(per)

    # Plafonne la part d'images négatives dans train (trop de négatives => rappel en baisse).
    labels = sorted((OUT / "train" / "labels").glob("*.txt"))
    neg = [p for p in labels if not p.read_text().strip()]
    pos = len(labels) - len(neg)
    max_neg = int(pos * args.max_neg_ratio / (1 - args.max_neg_ratio))
    if len(neg) > max_neg:
        drop = random.Random(0).sample(neg, len(neg) - max_neg)
        for lab in drop:
            for img in (OUT / "train" / "images").glob(lab.stem + ".*"):
                img.unlink()
            lab.unlink()
        stats["train"] -= len(drop)
        stats["négatives"] -= len(drop)
        print(f"\nNégatives en train plafonnées à {args.max_neg_ratio:.0%} : {len(drop)} retirées")

    has_test = any((OUT / "test" / "images").iterdir())
    cfg = {"path": str(OUT), "train": "train/images", "val": "valid/images",
           "names": {0: "japanese_beetle"}}
    if has_test:
        cfg["test"] = "test/images"
    YAML_OUT.write_text(
        "# Généré par merge_datasets.py — sources sous licence CC BY 4.0 (Roboflow Universe)\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    print(f"\nTotal : {stats['train']} train · {stats['valid']} valid · {stats['test']} test")
    print(f"        {stats['positives']} images avec scarabée(s), {stats['boîtes']} boîtes, "
          f"{stats['négatives']} images négatives, {stats['doublons']} doublons ignorés")
    print(f"Jeu écrit dans {OUT}\nÉtape suivante : .venv/bin/python train.py")


if __name__ == "__main__":
    main()

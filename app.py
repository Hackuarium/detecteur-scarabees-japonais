"""
Détecteur de scarabées japonais (Popillia japonica) — backend FastAPI + YOLOv8s.

Lancement :
    .venv/bin/python -m uvicorn app:app --port 8000

Poids du modèle :
    - Par défaut : models/japanese_beetle.pt (YOLOv8s ré-entraîné, voir train.py)
    - Sinon : variable d'environnement BEETLE_WEIGHTS=/chemin/vers/best.pt
    - En dernier recours : yolov8s.pt (COCO, mode démo — ne connaît PAS les scarabées)
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from torchvision.ops import nms
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = BASE_DIR / "models" / "japanese_beetle.pt"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
TILE_SIZE = 640
TILE_OVERLAP = 0.2


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model() -> tuple[YOLO, str, bool]:
    """Retourne (modèle, chemin des poids, mode_demo)."""
    env = os.environ.get("BEETLE_WEIGHTS")
    if env and Path(env).is_file():
        return YOLO(env), env, False
    if DEFAULT_WEIGHTS.is_file():
        return YOLO(str(DEFAULT_WEIGHTS)), str(DEFAULT_WEIGHTS), False
    # Repli : YOLOv8s pré-entraîné COCO (aucune classe "scarabée japonais").
    return YOLO("yolov8s.pt"), "yolov8s.pt (COCO)", True


MODEL, WEIGHTS_PATH, DEMO_MODE = _load_model()
DEVICE = _pick_device()
CLASS_NAMES: dict[int, str] = MODEL.names


def _beetle_class_ids() -> list[int] | None:
    """Classes à conserver. None = toutes (modèle mono-classe ou démo)."""
    if DEMO_MODE:
        return None
    ids = [i for i, n in CLASS_NAMES.items() if "beetle" in n.lower() or "scarab" in n.lower()]
    return ids or None


BEETLE_IDS = _beetle_class_ids()

app = FastAPI(title="Détecteur de scarabées japonais")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def _tiles(w: int, h: int, size: int, overlap: float):
    step = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(w - size, 0) + 1, step))
    ys = list(range(0, max(h - size, 0) + 1, step))
    if xs[-1] + size < w:
        xs.append(w - size)
    if ys[-1] + size < h:
        ys.append(h - size)
    for y in ys:
        for x in xs:
            yield x, y, min(x + size, w), min(y + size, h)


def _intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Aire d'intersection entre chaque boîte de a (N,4) et de b (M,4) -> (N,M)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    return (rb - lt).clamp(min=0).prod(dim=2)


def _predict(img: np.ndarray, conf: float, iou: float):
    res = MODEL.predict(img, conf=conf, iou=iou, imgsz=TILE_SIZE, device=DEVICE,
                        classes=BEETLE_IDS, verbose=False)[0]
    b = res.boxes
    return b.xyxy.cpu(), b.conf.cpu(), b.cls.cpu()


def detect(image: Image.Image, conf: float, iou: float, tiled: bool):
    arr = np.asarray(image)  # RGB
    arr = arr[:, :, ::-1].copy()  # -> BGR attendu par ultralytics pour les ndarray
    h, w = arr.shape[:2]

    if not tiled or (w <= TILE_SIZE * 1.25 and h <= TILE_SIZE * 1.25):
        boxes, scores, classes = _predict(arr, conf, iou)
        n_tiles = 1
    else:
        # Inférence par tuiles : les scarabées sont petits sur une photo de champ ;
        # redimensionner l'image entière à 640 px les ferait disparaître.
        # 1) Passe globale : voit en entier les gros sujets (gros plans).
        gb, gs, gc = _predict(arr, conf, iou)

        # 2) Passe par tuiles : trouve les petits insectes. On ignore toute détection
        #    qui touche un bord *intérieur* de tuile : c'est un insecte coupé, que la
        #    tuile voisine (chevauchement) ou la passe globale voit en entier.
        tb, ts, tc = [], [], []
        n_tiles = 0
        m = 3  # marge en px
        for x0, y0, x1, y1 in _tiles(w, h, TILE_SIZE, TILE_OVERLAP):
            n_tiles += 1
            b, s, c = _predict(arr[y0:y1, x0:x1], conf, iou)
            if not len(b):
                continue
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            cut = ((b[:, 0] <= x0 + m) & (x0 > 0)) | ((b[:, 1] <= y0 + m) & (y0 > 0)) \
                | ((b[:, 2] >= x1 - m) & (x1 < w)) | ((b[:, 3] >= y1 - m) & (y1 < h))
            keep = ~cut
            tb.append(b[keep]); ts.append(s[keep]); tc.append(c[keep])

        if tb and sum(len(x) for x in tb):
            tb, ts, tc = torch.cat(tb), torch.cat(ts), torch.cat(tc)
            # Doublons entre tuiles qui se chevauchent.
            k = nms(tb + tc[:, None] * (max(w, h) + 1), ts, iou)
            tb, ts, tc = tb[k], ts[k], tc[k]
            # 3) Priorité à la passe globale : on retire une détection de tuile si elle
            #    recouvre une détection globale (IoU) ou est contenue dedans (fragment).
            if len(gb):
                inter = _intersection(tb, gb)
                area_t = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
                area_g = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
                iou_m = inter / (area_t[:, None] + area_g[None, :] - inter + 1e-9)
                ios_m = inter / (area_t[:, None] + 1e-9)
                same = tc[:, None] == gc[None, :]
                dup = (((iou_m > iou) | (ios_m > 0.6)) & same).any(dim=1)
                tb, ts, tc = tb[~dup], ts[~dup], tc[~dup]
            boxes, scores, classes = torch.cat([gb, tb]), torch.cat([gs, ts]), torch.cat([gc, tc])
        else:
            boxes, scores, classes = gb, gs, gc

    dets = [
        {
            "box": [round(float(v), 1) for v in bx],
            "score": round(float(sc), 4),
            "class_id": int(cl),
            "label": CLASS_NAMES.get(int(cl), str(int(cl))),
        }
        for bx, sc, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist())
    ]
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets, n_tiles


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/info")
def info():
    return {
        "weights": WEIGHTS_PATH,
        "demo_mode": DEMO_MODE,
        "device": DEVICE,
        "classes": CLASS_NAMES,
        "beetle_class_ids": BEETLE_IDS,
    }


@app.post("/api/detect")
async def api_detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    tiled: bool = Form(True),
):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image trop volumineuse (max 30 Mo).")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")  # respecte l'orientation du téléphone
    except Exception:
        raise HTTPException(400, "Fichier image illisible.")

    conf = min(max(conf, 0.01), 0.99)
    iou = min(max(iou, 0.05), 0.95)

    t0 = time.perf_counter()
    dets, n_tiles = detect(img, conf, iou, tiled)
    elapsed = time.perf_counter() - t0

    return {
        "width": img.width,
        "height": img.height,
        "count": len(dets),
        "detections": dets,
        "tiles": n_tiles,
        "inference_ms": round(elapsed * 1000),
        "demo_mode": DEMO_MODE,
    }

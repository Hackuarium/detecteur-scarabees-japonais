# Détecteur de scarabées japonais (YOLOv8s)

Web app qui détecte *Popillia japonica* sur la photo d'une portion de champ.

- **Backend** : FastAPI + Ultralytics YOLOv8s (`app.py`)
- **Frontend** : page unique sans dépendance (`static/index.html`) — glisser-déposer, seuils réglables, boîtes dessinées, tableau des détections
- **Inférence par tuiles** : les photos haute résolution sont découpées en tuiles 640 px qui se chevauchent (20 %), puis les doublons sont fusionnés par NMS. Indispensable pour des insectes d'environ 1 cm sur une photo de champ.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Lancer l'app

```bash
.venv/bin/python -m uvicorn app:app --port 8000
```

Puis ouvrir http://localhost:8000.

## Important : il faut un modèle entraîné

YOLOv8s « de base » est entraîné sur COCO (80 classes : personne, voiture, chien…) et **ne connaît pas les scarabées japonais**. Sans poids spécifiques, l'app démarre en *mode démo* et l'indique.

1. Exporter au format **YOLOv8** (ZIP) les jeux Roboflow Universe suivants (CC BY 4.0) et les déposer dans `downloads/` :
   - [Japanese Beetle Identification](https://universe.roboflow.com/insect-identification-dgace/japanese-beetle-identification) — 1 091 images (pas de version publiée : « Fork Dataset », générer une version, puis exporter)
   - [InsectDetectionV2](https://universe.roboflow.com/cool-internship/insectdetectionv2) — 537 images, 10 insectes
   - [pest-demo](https://universe.roboflow.com/aditya-sengupta/pest-demo) — 156 images, 4 ravageurs

   Vos propres photos annotées (export YOLO) peuvent aussi y aller.
2. Fusionner en un jeu mono-classe (les autres insectes deviennent des exemples négatifs) :
   ```bash
   .venv/bin/python merge_datasets.py
   ```
3. Entraîner :
   ```bash
   .venv/bin/python train.py --epochs 100
   ```
   Le meilleur modèle est copié dans `models/japanese_beetle.pt` et chargé automatiquement au prochain démarrage. On peut aussi pointer vers d'autres poids : `BEETLE_WEIGHTS=/chemin/best.pt`.

Sur un Mac Apple Silicon, l'entraînement et l'inférence utilisent le GPU via MPS.

## API

`POST /api/detect` (multipart) : `file`, `conf` (0.25), `iou` (0.45), `tiled` (true)
→ `{count, detections:[{box:[x1,y1,x2,y2], score, label}], width, height, tiles, inference_ms, demo_mode}`

`GET /api/info` → poids chargés, classes, appareil, mode démo.

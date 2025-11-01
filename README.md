# veg_reg_yolov8

A small, opinionated repository for vegetable detection/registration using YOLOv8.
This project includes a pretrained `model.pt`, a simple Flask-based web UI (in `main.py` + `templates/` + `static/`), and example dataset configuration (`data.yaml`).

## Highlights

- Lightweight inference and a minimal web UI for quick demos.
- Uses Ultralytics YOLOv8 under the hood (see `requirements.txt`).
- Included `model.pt` for immediate local testing.

## Requirements

- Python 3.8+
- See `requirements.txt` for required Python packages. Primary dependencies:

  # veg_reg_yolov8

  An easy-to-run demo for vegetable detection using YOLOv8 and a small Flask web UI.

  This repository contains:

  - `model.pt` — pretrained YOLOv8 weights (for demo/inference).
  - `main.py` — minimal Flask app that serves a small web UI and exposes an upload API.
  - `templates/`, `static/` — frontend HTML/CSS/JS for the UI.
  - `data.yaml` — dataset configuration (class names, train/val paths).

  ## Features

  - Live webcam stream with per-class counts (real-time tracking).
  - Image upload API that returns annotated image and per-class counts.
  - Simple, responsive UI with separate pages for Live and Upload.

  ## Requirements

  - Python 3.8+
  - See `requirements.txt` for full dependency list. Key packages:
    - `ultralytics` (YOLOv8)
    - `opencv-python`
    - `flask`

  Install and activate a virtual environment, then install dependencies:

  ```bash
  python -m venv .venv
  source .venv/bin/activate  # macOS / Linux (zsh)
  pip install -r requirements.txt
  ```

  ## Run the app (local)

  1. Make sure `model.pt` is present in the repository root. The Flask app expects it there.
  2. Start the server:

  ```bash
  python main.py
  ```

  3. Open your browser:

  - Landing page: http://127.0.0.1:5000
  - Live stream: http://127.0.0.1:5000/live
  - Upload page: http://127.0.0.1:5000/upload

  Notes:

  - The live page reads from the system default webcam (index 0). If your webcam is on a different index or in use, change the index in `main.py` (`cv2.VideoCapture(0)`).
  - If you want to run on a different host/port, set environment variables or edit the call to `app.run()` in `main.py`.

  ## API

  - `GET /` — landing page with links to Live and Upload pages.
  - `GET /live` — page showing live stream and counts panel.
  - `GET /upload` — web page that contains the upload form (for users).
  - `POST /upload` — accepts multipart form `file` (image). Response JSON:

  ```json
  {
    "counts": { "tomato": 2, "onion": 1 },
    "image": "data:image/jpeg;base64,..."
  }
  ```

  - `GET /video_feed` — MJPEG stream for the live camera (used by `/live`).
  - `GET /counts` — JSON with latest per-class counts from the live tracker.

  ## Data & labels

  `data.yaml` contains class names and dataset paths in YOLO format. Example entry for the names used in this project:

  ```yaml
  names:
    [
      "avocado",
      "beans",
      "beet",
      "bell pepper",
      "broccoli",
      "brus capusta",
      "cabbage",
      "carrot",
      "cayliflower",
      "celery",
      "corn",
      "cucumber",
      "eggplant",
      "fasol",
      "garlic",
      "hot pepper",
      "onion",
      "peas",
      "potato",
      "pumpkin",
      "rediska",
      "redka",
      "salad",
      "squash-patisson",
      "tomato",
      "vegetable marrow",
    ]
  ```

  When training a new model, make sure `data.yaml` points to the correct `train` and `val` image folders and has the correct `nc` (number of classes).

  ## Troubleshooting

  - Webcam not found: check that your webcam is free and the index in `cv2.VideoCapture(...)` is correct.
  - Model file missing: ensure `model.pt` exists at the repo root.
  - Slow inference on CPU: try a smaller model (e.g., `yolov8n.pt`) or run on a machine with GPU and install matching PyTorch/CUDA.
  - Dependency issues: re-create the virtualenv and `pip install -r requirements.txt`.

  ## Development notes

  - To change UI text or add localization, edit the templates in `templates/` and static assets in `static/`.
  - Consider moving inline JavaScript into `static/` files for easier maintenance.
  - Add a systemd/service file or Dockerfile if you plan to deploy.

  ## License

  Add a LICENSE file to this repository if you want an explicit license (e.g., MIT). The current repository is provided as-is for demo and development.

  ## Contact

  Open an issue in the repository if you need help, or contact the maintainer directly.

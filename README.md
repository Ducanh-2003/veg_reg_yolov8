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
  - `ultralytics` (YOLOv8)
  - `opencv-python`
  - `flask`

Install dependencies with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux (zsh)
pip install -r requirements.txt
```

## Quickstart — run the web UI

The repository contains a minimal Flask web app for running inference locally.

1. Ensure your virtualenv is active and dependencies installed.
2. Start the app:

```bash
python main.py
```

3. Open your browser to http://127.0.0.1:5000 and use the UI to upload images and view detections.

(If `main.py` offers CLI flags the app may accept `--host` / `--port` overrides.)

## Quickstart — command-line inference (Ultralytics)

If you prefer the Ultralytics CLI for quick inference, use:

```bash
# single image or folder
yolo detect predict model=model.pt source=path/to/image_or_folder
```

This will run detection and write result images to `runs/detect/...` by default.

## Training

Train a new model with the Ultralytics `yolo` command. Example:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

- `data.yaml` should point to your train/val paths and class names (YOLO format).
- For more advanced training, adjust hyperparameters or use a larger backbone (e.g., `yolov8s.pt`, `yolov8m.pt`).

## Dataset format

This project uses the standard YOLO text-label format:

- Images: any supported image format (JPEG/PNG) under your dataset folders.
- Labels: one `.txt` file per image with lines: `<class> <x_center> <y_center> <width> <height>` — normalized (0..1).
- `data.yaml` example:

```yaml
train: /path/to/train/images
val: /path/to/val/images
nc: 3
names: ['avocado', 'beans', 'beet', 'bell pepper', 'broccoli', 'brus capusta', 'cabbage', 'carrot', 'cayliflower', 'celery', 'corn', 'cucumber', 'eggplant', 'fasol', 'garlic', 'hot pepper', 'onion', 'peas', 'potato', 'pumpkin', 'rediska', 'redka', 'salad', 'squash-patisson', 'tomato', 'vegetable marrow']
```

## Included artifacts

- `model.pt` — a pretrained weights file included for quick demos.
- `main.py` — the minimal Flask server / inference entrypoint.
- `data.yaml` — dataset configuration used for training/inference.
- `templates/`, `static/` — simple web UI assets.

## Troubleshooting

- If you see missing package errors, ensure your virtualenv is activated and `pip install -r requirements.txt` completed without errors.
- If inference is slow on CPU, try smaller model sizes (e.g., `yolov8n.pt`) or run on GPU (install appropriate PyTorch + CUDA where available).
- If the web UI doesn't start, check for port collisions or run `python main.py --port 8000` (if supported) or set the `FLASK_APP` env var as needed.

## Development notes

- To add a new class, update your dataset labels and `data.yaml`, then retrain a model.
- Consider adding CI checks for linting or a smoke test that runs a single inference on a tiny sample image.

## License

This project is provided as-is. Add a license file if you want an explicit license (e.g., MIT).

## Contact

If you have questions or need help, open an issue in the repository or contact the maintainer.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fix OMP duplicate lib issue (keep)

import cv2
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO
import threading
import numpy as np
import base64

# --- INITIALIZE FLASK APPLICATION ---
app = Flask(__name__)

# --- LOAD MODEL AND WEBCAM ---
# Load YOLOv8 model (expect model.pt in repo root)
model = YOLO('model.pt') 

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# Global to store latest counts for the web UI
latest_counts = {}
counts_lock = threading.Lock()


def generate_frames():
    """
    Generator that continuously reads from the webcam,
    runs the model, annotates the frame and yields JPEG frames
    for an MJPEG response.
    """
    while True:
        # 1. Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            print("No frame or webcam error.")
            break

        # 2. Run the tracker / model
        # conf=0.25 gives stable tracking
        results = model.track(frame, persist=True, conf=0.25, verbose=False)

        # Get the annotated frame (boxes, confs, etc.)
        annotated_frame = results[0].plot(boxes=True, masks=False, conf=True)

        # Extract confs (confidence), class ids and tracker ids
        boxes = results[0].boxes
        confs = []
        cls_ids = []
        track_ids = []
        try:
            if boxes.conf is not None:
                confs = boxes.conf.cpu().tolist()
        except Exception:
            confs = []
        try:
            if boxes.cls is not None:
                # boxes.cls may be float tensor; convert to int
                cls_ids = [int(x) for x in boxes.cls.cpu().tolist()]
        except Exception:
            cls_ids = []
        try:
            if boxes.id is not None:
                track_ids = [int(x) for x in boxes.id.int().cpu().tolist()]
        except Exception:
            track_ids = []

        # Ensure lists are same length; if ids missing, use None placeholders
        n = max(len(confs), len(cls_ids), len(track_ids))
        # pad lists
        while len(confs) < n:
            confs.append(0.0)
        while len(cls_ids) < n:
            cls_ids.append(-1)
        while len(track_ids) < n:
            track_ids.append(None)

        # Threshold for considering a detection as "counted"
        CONF_THRESH = 0.8

        # Map track_id -> class_id for high-confidence detections
        tid_to_cid = {}
        for conf, cid, tid in zip(confs, cls_ids, track_ids):
            if tid is not None and conf >= CONF_THRESH:
                tid_to_cid[tid] = cid

        # If tracker didn't return ids, fallback to counting detections by class
        counts = {}
        if len(tid_to_cid) > 0:
            # Count unique track ids grouped by class
            for tid, cid in tid_to_cid.items():
                # Resolve name from model.names if possible
                name = model.names.get(cid, str(cid)) if hasattr(model, 'names') else str(cid)
                counts[name] = counts.get(name, 0) + 1
        else:
            # Fallback: count detections with conf >= CONF_THRESH by class
            for conf, cid in zip(confs, cls_ids):
                if conf >= CONF_THRESH and cid >= 0:
                    name = model.names.get(cid, str(cid)) if hasattr(model, 'names') else str(cid)
                    counts[name] = counts.get(name, 0) + 1

        # Update global latest_counts for the web UI
        with counts_lock:
            latest_counts.clear()
            # copy so we don't hold references to same dict
            for k, v in counts.items():
                latest_counts[k] = v

        # total count of detected objects (high confidence)
        current_object_count = sum(latest_counts.values())

        # overlay a readable counter on the frame
        cv2.putText(annotated_frame, f"COUNT: {current_object_count}", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 175, 76), 2)

        # Optionally draw per-class counts on frame (top-left), multiple lines
        start_y = 110
        line_h = 28
        i = 0
        for name, cnt in latest_counts.items():
            text = f"{name}: {cnt}"
            cv2.putText(annotated_frame, text, (50, start_y + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            i += 1

        # 4. Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue  # Skip if encoding fails

        # 5. Stream the frame
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html') # Trang chính (landing) links tới Live và Upload


@app.route('/live')
def live():
    """Serve the live webcam page."""
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts_api():
    """Return latest per-class counts as JSON."""
    with counts_lock:
        # Return a copy to avoid race
        data = dict(latest_counts)
    return jsonify({'counts': data})

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    """If GET: return the upload page. If POST: accept image, run detection and return JSON."""
    if request.method == 'GET':
        return render_template('upload.html')

    # POST: API xử lý ảnh
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    data = file.read()
    # Convert bytes to numpy image
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Cannot decode image'}), 400

    # Run detection (single-frame predict)
    results = model.predict(img, conf=0.25, verbose=False)
    annotated = results[0].plot(boxes=True, masks=False, conf=True)

    # Count detections by class (no track ids for single image)
    counts = {}
    boxes = results[0].boxes
    try:
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    except Exception:
        confs = []
    try:
        cls_ids = [int(x) for x in boxes.cls.cpu().tolist()] if boxes.cls is not None else []
    except Exception:
        cls_ids = []

    # pad
    n = max(len(confs), len(cls_ids))
    while len(confs) < n:
        confs.append(0.0)
    while len(cls_ids) < n:
        cls_ids.append(-1)

    CONF_THRESH = 0.5
    for conf, cid in zip(confs, cls_ids):
        if conf >= CONF_THRESH and cid >= 0:
            name = model.names.get(cid, str(cid)) if hasattr(model, 'names') else str(cid)
            counts[name] = counts.get(name, 0) + 1

    # Encode annotated image to base64
    ret, buffer = cv2.imencode('.jpg', annotated)
    if not ret:
        return jsonify({'error': 'Encoding failed'}), 500
    jpg_bytes = buffer.tobytes()
    b64 = base64.b64encode(jpg_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{b64}"

    return jsonify({'counts': counts, 'image': data_url})

# --- CHẠY SERVER ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
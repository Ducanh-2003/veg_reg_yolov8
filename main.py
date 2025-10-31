import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Sửa lỗi OMP (giữ lại)

import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO

# --- KHỞI TẠO ỨNG DỤNG FLASK ---
app = Flask(__name__)

# --- TẢI MODEL VÀ WEBCAM ---
# Tải model YOLOv8 (file .pt phải ở cùng thư mục)
model = YOLO('model.pt') 

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở webcam.")
    exit()

def generate_frames():
    """
    Hàm này là một "generator", nó liên tục đọc webcam,
    chạy model, và "phát" (yield) từng frame ra ngoài.
    """

    
    while True:
        # 1. Đọc frame từ webcam
        success, frame = cap.read()
        if not success:
            print("Hết frame hoặc lỗi webcam.")
            break
        
        # 2. CHẠY MODEL TRACKING
        # conf=0.25 để tracker chạy ổn định
        results = model.track(frame, persist=True, conf=0.25, verbose=False)

        # Lấy frame đã được vẽ (boxes, confs...)
        annotated_frame = results[0].plot(boxes=True, masks=False, conf=True)

        
        # Tạo một set để lưu các ID của các vật thể có conf > 0.9 trong frame
        current_high_conf_ids = set()

        # Lấy confs (độ tự tin) và tracker_ids
        confs = results[0].boxes.conf.cpu()
        track_ids = []
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()

        # Lặp qua từng đối tượng
        for conf, track_id in zip(confs, track_ids):
            
            if conf > 0.8:
                current_high_conf_ids.add(track_id) 

        # counter
        current_object_count = len(current_high_conf_ids)

        # counter text
        cv2.putText(annotated_frame, f"SO LUONG: {current_object_count}", (50, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 175, 76), 2)

        # 4. Mã hóa frame thành JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue # Bỏ qua nếu nén lỗi

        # 5. "Phát" frame ra
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
   return render_template('index.html') # Đảm bảo file HTML ở thư mục /templates

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- CHẠY SERVER ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
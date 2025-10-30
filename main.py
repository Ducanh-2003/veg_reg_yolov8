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
        
        # 2. Chạy model YOLOv8 trên frame
        # stream=True sẽ hiệu quả hơn cho video
        results = model(frame, task='detect', conf=0.25, verbose=False, stream=True)

        # 3. Lấy frame đã được vẽ
        annotated_frame = frame # Mặc định là frame gốc
        for r in results:
            annotated_frame = r.plot(boxes=True, masks=False) # Chỉ vẽ box

        # 4. Mã hóa frame thành JPEG
        # .imencode sẽ nén ảnh thành định dạng JPEG trong bộ nhớ
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue # Bỏ qua nếu nén lỗi

        # 5. "Phát" frame ra
        # Chuyển buffer thành bytes và gửi đi theo chuẩn HTTP
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN (ROUTES) ---

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Đây là đường dẫn "nguồn" video.
    Trình duyệt sẽ liên tục kết nối tới đây để lấy video.
    Nó trả về một 'Response' kiểu 'multipart/x-mixed-replace',
    cho phép server liên tục "thay thế" ảnh cũ bằng ảnh mới.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- CHẠY SERVER ---
if __name__ == '__main__':
    # 'debug=True' để server tự khởi động lại khi bạn sửa code
    # 'host='0.0.0.0'' để server có thể được truy cập từ máy khác trong mạng
    app.run(host='0.0.0.0', port=5000, debug=True)
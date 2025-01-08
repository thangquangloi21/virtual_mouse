# Import Libraries
import cv2
import time
import mediapipe as mp


# Lấy Mô hình Toàn diện từ Mediapipe và
# Khởi tạo Mô hình

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)

# Khởi tạo các tiện ích vẽ để vẽ các điểm mốc trên khuôn mặt trên hình ảnh
mp_drawing = mp.solutions.drawing_utils
# (0) trong VideoCapture được sử dụng để kết nối với camera mặc định của máy tính của bạn
capture = cv2.VideoCapture(0)

# Khởi tạo thời gian hiện tại và thời gian quý báu để tính FPS
previousTime = 0
currentTime = 0

while capture.isOpened():
    # chụp từng khung hình
    ret, frame = capture.read()

    # thay đổi kích thước khung hình để xem tốt hơn
    frame = cv2.resize(frame, (800, 600))

    # Chuyển đổi từ BGR sang RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Đưa ra dự đoán bằng mô hình toàn diện
    # Để cải thiện hiệu suất, tùy chọn đánh dấu hình ảnh là không thể ghi vào
    # truyền theo tham chiếu.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Chuyển đổi lại hình ảnh RGB sang BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Vẽ các điểm mốc trên khuôn mặt
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(
            color=(255, 0, 255),
            thickness=1,
            circle_radius=1
        ),
        mp_drawing.DrawingSpec(
            color=(0, 255, 255),
            thickness=1,
            circle_radius=1
        )
    )

    # Vẽ các mốc bên tay phải
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    # Vẽ các mốc bằng tay trái
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    # Tính toán FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Hiển thị FPS trên hình ảnh
    cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị hình ảnh kết quả
    cv2.imshow("Facial and Hand Landmarks", image)

    # Mã để truy cập các địa danh
    for landmark in mp_holistic.HandLandmark:
        print(landmark, landmark.value)

    print(mp_holistic.HandLandmark.WRIST.value)


    # Nhập phím 'q' để phá vòng lặp
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Khi tất cả quá trình hoàn tất
# Giải phóng capture và hủy tất cả các cửa sổ
capture.release()
cv2.destroyAllWindows()


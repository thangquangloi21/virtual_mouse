import csv
import time
import cv2
import mediapipe as mp

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo mô hình Mediapipe
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarks_data = []
video = "Video_test/tayphai.mp4"
# Mở video
capture = cv2.VideoCapture(video)

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (800, 600))  # Resize khung hình
    if not ret:
        break

    # Chuyển đổi từ BGR sang RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Dự đoán bàn tay
    results = hands_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Kiểm tra nếu có bàn tay được phát hiện
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Lấy thông tin tay trái hoặc phải
            hand_label = results.multi_handedness[idx].classification[0].label

            # Chỉ xử lý tay phải
            if hand_label == "Right":
                h, w, c = image.shape
                frame_landmarks = []  # Lưu các điểm của bàn tay trong khung hình
                for landmark in hand_landmarks.landmark:
                    x, y, z = landmark.x * w, landmark.y * h, landmark.z
                    frame_landmarks.append((x, y, z))
                landmarks_data.append(frame_landmarks)

                # Vẽ toàn bộ bàn tay
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị hình ảnh kết quả
    cv2.imshow("Hand Tracking with Mouse Control", image)

    # Nhập phím 'q' để phá vòng lặp
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Xuất dữ liệu ra file CSV
with open('hand_landmarks.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Landmark_Index", "X", "Y", "Z"])  # Tiêu đề
    for frame_idx, frame_landmarks in enumerate(landmarks_data):
        for landmark_idx, (x, y, z) in enumerate(frame_landmarks):
            writer.writerow([frame_idx, landmark_idx, x, y, z])

print("Dữ liệu đã được xuất ra file hand_landmarks.csv.")

# Giải phóng tài nguyên
capture.release()
cv2.destroyAllWindows()

import csv
import os
import pandas as pd
import cv2
import mediapipe as mp

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo mô hình Mediapipe
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Xử lý cả 2 tay
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_hand_landmarks(hand_landmarks):
    """Hàm trích xuất tọa độ các điểm trên bàn tay."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append(lm.x)
        landmarks.append(lm.y)
        landmarks.append(lm.z)
    return landmarks

landmarks_data = []
# Đường dẫn video và đầu ra
video = "D:\\virtual_mouse-test_branch\\virtual_mouse-test_branch\\Video_test\\Start\\start1.mp4"
output = "D:\\virtual_mouse-test_branch\\virtual_mouse-test_branch\\Data\\Start"

# Mở video
capture = cv2.VideoCapture(video)
if not capture.isOpened():
    print("Không thể mở video. Kiểm tra đường dẫn hoặc định dạng tệp.")
    exit()

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        print("Không đọc được khung hình. Kết thúc hoặc video đã phát xong.")
        break

    # Kiểm tra và resize khung hình
    try:
        frame = cv2.resize(frame, (800, 600))
    except cv2.error as e:
        print(f"Lỗi khi resize khung hình: {e}")
        continue

    # Chuyển đổi từ BGR sang RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Dự đoán bàn tay
    results = hands_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Kiểm tra nếu có bàn tay được phát hiện
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label

            # Xử lý tay phải
            if hand_label == "Right":
                h, w, c = image.shape
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                # Trích xuất tọa độ bàn tay
                hand_frame = extract_hand_landmarks(hand_landmarks)
                landmarks_data.append(hand_frame)

                # Vẽ điểm ngón chỏ
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(image, f"Right Hand({x},{y})", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Vẽ toàn bộ bàn tay
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Xử lý tay trái
            if hand_label == "Left":
                h, w, c = image.shape
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                # Trích xuất tọa độ bàn tay
                hand_frame = extract_hand_landmarks(hand_landmarks)
                landmarks_data.append(hand_frame)

                # Vẽ điểm ngón chỏ
                cv2.circle(image, (x, y), 10, (255, 0, 0), -1)
                cv2.putText(image, f"Left Hand({x},{y})", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Vẽ toàn bộ bàn tay
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị hình ảnh kết quả
    cv2.imshow("Hand Tracking with Mouse Control", image)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lưu dữ liệu landmarks vào tệp CSV
if len(landmarks_data) > 0:
    columns = [str(i) for i in range(len(landmarks_data[0]))]
    df = pd.DataFrame(landmarks_data, columns=columns)
    csv_path = os.path.join(output, "start_mouse1.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dữ liệu đã được lưu thành công tại {csv_path}")

# Giải phóng tài nguyên
capture.release()
cv2.destroyAllWindows()

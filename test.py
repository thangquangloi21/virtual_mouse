import time
import cv2
import mediapipe as mp
import pyautogui

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pyautogui.FAILSAFE = False

# Khởi tạo mô hình Mediapipe
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)

# Lưu vị trí ngón tay trước đó
previous_x, previous_y = None, None

# Hệ số khuếch đại
amplification_factor = 10

# Mở camera
capture = cv2.VideoCapture(0)

# Biến đếm và thời gian
count = 0
start_time = time.time()

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
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y, z = int(index_tip.x * w), int(index_tip.y * h), int(index_tip.z * (-100))

                # Tăng biến đếm khi phát hiện tọa độ
                count += 1

                # Cập nhật vị trí trước đó
                if previous_x is not None and previous_y is not None:
                    delta_x = (x - previous_x) * amplification_factor
                    delta_y = (y - previous_y) * amplification_factor
                    pyautogui.moveRel(delta_x, delta_y)

                previous_x, previous_y = x, y

                # Vẽ điểm ngón tay trên khung hình
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(image, f"Right Hand({x},{y})", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Vẽ toàn bộ bàn tay
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        previous_x, previous_y = None, None

    # Hiển thị hình ảnh kết quả
    cv2.imshow("Hand Tracking with Mouse Control", image)

    # Kết thúc đếm sau 1 giây
    if time.time() - start_time >= 1:
        continue

    # Nhập phím 'q' để phá vòng lặp
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# In kết quả số lần đếm
print(f"Số lần tọa độ được cập nhật trong 1 giây: {count}")

# Giải phóng tài nguyên
capture.release()
cv2.destroyAllWindows()

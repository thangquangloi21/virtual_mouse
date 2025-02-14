import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
import pydirectinput
from pynput.mouse import Controller, Button
import mouse

# Tắt chế độ failsafe của pyautogui và pydirectinput
pyautogui.FAILSAFE = False
pydirectinput.FAILSAFE = False

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Tải model (dùng raw string để tránh lỗi escape)
model = tf.keras.models.load_model(r"D:\virtual_mouse-test_branch\virtual_mouse-test_branch\MODEL\model_lstm.keras")
labels = {
    0: "Move",
    1: "Pause",
    2: "Scroll",
    3: "Start",
}
lm_list = []

# Khởi tạo mô hình Hands của Mediapipe
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mouse = Controller()
screenWidth, screenHeight = pyautogui.size()

previous_x, previous_y = None, None  # Lưu vị trí chuột trước đó
prev_index_finger_y = None  # Lưu vị trí y của ngón trỏ tay trái (cho chế độ scroll)

# Hệ số khuếch đại cho chuyển động chuột
amplification_factor = 20

# Biến mode để điều chỉnh hành vi: "move" hay "scroll"
mode = "move"  # mặc định là di chuyển chuột

# Biến để kiểm soát trạng thái Start/Pause
is_active = False  # Mặc định là không hoạt động


def detect(model, lm_list):
    """Nhận diện cử chỉ dựa trên chuỗi landmark."""
    data = np.expand_dims(np.array(lm_list), axis=0)
    results = model.predict(data)
    return np.argmax(results, axis=1)[0], np.max(results)


def make_landmark_timestep(hand_landmarks):
    """Chuyển đổi landmark của một bàn tay thành danh sách các giá trị [x, y, z]."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks


def get_hand_landmark(hand_landmarks):
    """Trả về danh sách tọa độ x và y từ landmark của bàn tay."""
    x_list = [lm.x for lm in hand_landmarks.landmark]
    y_list = [lm.y for lm in hand_landmarks.landmark]
    return x_list, y_list


def draw_rectangle(hand, image, W, H, text):
    """Vẽ hình chữ nhật bao quanh bàn tay và in text lên đó."""
    x_list, y_list = get_hand_landmark(hand)
    x1 = int(min(x_list) * W) - 10
    y1 = int(min(y_list) * H) - 10
    x2 = int(max(x_list) * W) + 10
    y2 = int(max(y_list) * H) + 10
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(image, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)


# Các tham số cho nhận diện cử chỉ
i = 0
warmup_frames = 60
sequence_length = 10
predicted_action = ""
predicted_action_index = 0

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Lật và thay đổi kích thước khung hình
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (800, 600))
    H, W, _ = frame.shape

    # Chuyển đổi màu để xử lý với Mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands_model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    i += 1
    if i > warmup_frames:
        left_hand, right_hand = None, None

        # Phân loại tay trái và tay phải nếu có
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    right_hand = hand_landmarks
                elif handedness.classification[0].label == "Left":
                    left_hand = hand_landmarks

        # Cập nhật chuỗi landmark cho dự đoán

        if right_hand:
            lm_list.append(make_landmark_timestep(right_hand))
        elif left_hand:
            lm_list.append(make_landmark_timestep(left_hand))

        if len(lm_list) > sequence_length:
            lm_list = lm_list[-sequence_length:]
        if len(lm_list) == sequence_length:
            prediction, percentage = detect(model, lm_list)
            if percentage >= 0.9:
                predicted_action = labels.get(prediction, "")
                predicted_action_index = prediction
                # Cập nhật chế độ dựa theo dự đoán từ model
                if predicted_action_index == 2:
                    mode = "scroll"
                elif predicted_action_index == 0:
                    mode = "move"
                elif predicted_action_index == 3:  # Start
                    is_active = True
                    print("Start: Active")
                elif predicted_action_index == 1:  # Pause
                    is_active = False
                    print("Pause: Inactive")
                print("Predicted:", predicted_action)
            lm_list = []

        # Xử lý tay trái
        if left_hand and is_active:  # Chỉ xử lý nếu đang ở trạng thái Active
            if mode == "move":
                # Sử dụng if/elif để tránh click chồng (chỉ thực hiện 1 click mỗi frame)
                if left_hand.landmark[8].y > left_hand.landmark[7].y:
                    mouse.click(Button.left)
                    print("Left Click Performed")
            elif mode == "scroll":
                # Dùng vị trí ngón trỏ để cuộn trang
                index_finger = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_x = int(index_finger.x * W)
                index_finger_y = int(index_finger.y * H)
                cv2.putText(image, f"Index Finger: ({index_finger_x}, {index_finger_y})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if prev_index_finger_y is not None:
                    delta_y = index_finger_y - prev_index_finger_y
                    if delta_y < -10:
                        pyautogui.scroll(50)
                        print("Scrolling Up")
                    elif delta_y > 10:
                        pyautogui.scroll(-50)
                        print("Scrolling Down")
                prev_index_finger_y = index_finger_y
        else:
            prev_index_finger_y = None  # Reset nếu tay trái không xuất hiện

        # Xử lý tay phải (di chuyển chuột) chỉ khi đang ở mode "move" và is_active
        if right_hand and mode == "move" and is_active:
            x, y, z = int(right_hand.landmark[8].x * W), int(right_hand.landmark[8].y * H), int(
                right_hand.landmark[8].z * (-100))
            x1 = (round(x / 10) * 10) / 10
            y1 = (round(y / 10) * 10) / 10
            if previous_x is not None and previous_y is not None:
                delta_x = (x1 - previous_x) * amplification_factor / (W / screenWidth)
                delta_y = (y1 - previous_y) * amplification_factor / (H / screenHeight)
                try:
                    mouse.move(delta_x, delta_y)
                    print(f"Moving mouse by ({delta_x}, {delta_y})")
                except Exception as error:
                    print("Error moving mouse:", error)
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(image, f"Right Hand({x},{y})", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            previous_x, previous_y = x1, y1
        else:
            previous_x, previous_y = None, None

        # Vẽ landmark và bounding box cho các tay
        if left_hand:
            mp_drawing.draw_landmarks(image, left_hand, mp_hands.HAND_CONNECTIONS)
        if right_hand:
            mp_drawing.draw_landmarks(image, right_hand, mp_hands.HAND_CONNECTIONS)
            draw_rectangle(right_hand, image, W, H, predicted_action)

    cv2.imshow("Hand Tracking with Mouse Control", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
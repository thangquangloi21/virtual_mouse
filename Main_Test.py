import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model = tf.keras.models.load_model("D:\All_learn_programs\Python\\virtualMouse\MODEL\model_rnn.h5")
labels = {0:"Move Mouse", 1: "Left Mouse", 2:"RightMouse",3:"ScrollMouse"}
lm_list = []
# Khởi tạo mô hình Mediapipe
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect(model, lm_list):
    lm_list = np.expand_dims(np.array(lm_list), axis=0)
    results = model.predict(lm_list)
    return np.argmax(results, axis=1)
def make_landmark_timestep(results):
    landmarks = []
    for id, lm in enumerate(results.landmark):
        landmarks.append(lm.x)
        landmarks.append(lm.y)
        landmarks.append(lm.z)
    return landmarks
def get_hand_landmark(results):
    x_list = []
    y_list =[]
    for lm in results.landmark:
        x_list.append(lm.x)
        y_list.append(lm.y)
    return x_list,y_list

i = 0
warmup_frames = 60
sequence_length = 10
predicted_action = ""

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    # resize khung hình
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (600, 400))
    H, W, _ = frame.shape
    # Chuyển đổi từ BGR sang RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Dự đoán bàn tay
    results = hands_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    i += 1
    x_ = []
    y_ = []
    if i > warmup_frames:
        print("Start detect . . .")
        # Kiểm tra nếu có bàn tay được phát hiện
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Lấy thông tin tay trái hoặc phải
                hand_label = results.multi_handedness[idx].classification[0].label

                # Chỉ xử lý tay phải
                if hand_label == "Right":
                    x_, y_ = get_hand_landmark(hand_landmarks)
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10

                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(image, predicted_action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    frame_landmarks = make_landmark_timestep(hand_landmarks)
                    lm_list.append(frame_landmarks)
                    if len(lm_list) > sequence_length:
                        lm_list = lm_list[-sequence_length:]
                    if len(lm_list) == sequence_length:
                        prediction = detect(model, lm_list)
                        predicted_action = labels.get(prediction[0])
                        print("Action Id: ", prediction[0])
                        print("Action Detected:", predicted_action)
                        # cv2.putText(image, predicted_action, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        lm_list = []  # Reset list after prediction
                    # Vẽ toàn bộ bàn tay
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # Hiển thị hình ảnh kết quả
    cv2.imshow("Hand Tracking with Mouse Control", image)
    # Nhập phím 'q' để phá vòng lặp
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
capture.release()
cv2.destroyAllWindows()
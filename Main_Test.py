import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf
import mouse

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained model
model = tf.keras.models.load_model("D:\All_learn_programs\Python\\virtualMouse\MODEL\model_gru.h5")
labels = {0: "LeftMouse",1: "MoveMouse", 2: "RightMouse", 3: "ScrollDown", 4: "ScrollUp",5: "ZoomIn", 6: "Zoomout",7: "Off", 8: "Start", 9: "PauseCursor"}
lm_list = []

# Initialize Mediapipe hands model
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Xử lý cả hai tay
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# Screen dimensions
screen_width, screen_height = mouse.get_position()

# Smooth movement function
def smooth_move(target_x, target_y, duration=0.1, steps=10):
    current_x, current_y = mouse.get_position()
    step_x = (target_x - current_x) / steps
    step_y = (target_y - current_y) / steps
    for i in range(steps):
        current_x += step_x
        current_y += step_y
        mouse.move(int(current_x), int(current_y))
        time.sleep(duration / steps)

def detect(model, lm_list):
    lm_list = np.expand_dims(np.array(lm_list), axis=0)
    results = model.predict(lm_list)
    print(results)
    return np.argmax(results, axis=1), np.max(results)

def make_landmark_timestep(results):
    landmarks = []
    for lm in results.landmark:
        landmarks.append(lm.x)
        landmarks.append(lm.y)
        landmarks.append(lm.z)
    return landmarks

def get_hand_landmark(hand_landmarks):
    x_list = []
    y_list = []
    for lm in hand_landmarks.landmark:
        x_list.append(lm.x)
        y_list.append(lm.y)
    return x_list, y_list

# Parameters
i = 0
warmup_frames = 60
sequence_length = 10
predicted_action = ""
# Start video capture
capture = cv2.VideoCapture(0)
accept_action = False
predicted_action_index = 0
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Flip and resize frame
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    H, W, _ = frame.shape

    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process hand landmarks
    results = hands_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    i += 1
    if i > warmup_frames and i % 3 == 0:
        if results.multi_hand_landmarks:
            left_hand, right_hand = None, None

            # Phân loại tay trái và tay phải
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                if handedness.classification[0].label == "Right":
                    right_hand = hand_landmarks
                elif handedness.classification[0].label == "Left":
                    left_hand = hand_landmarks

            if len(lm_list) > sequence_length:
                lm_list = lm_list[-sequence_length:]

            if len(lm_list) == sequence_length:
                prediction, percentage = detect(model, lm_list)
                if percentage >= 0.9:
                    predicted_action = labels.get(prediction[0])
                    predicted_action_index = prediction[0]
                lm_list = []


            # Nếu cả hai tay cùng xuất hiện
            if left_hand and right_hand:
                lm_list.append(make_landmark_timestep(left_hand))
                lm_list.append(make_landmark_timestep(right_hand))

                # Xử lý "Start" và "Off"
                if predicted_action_index == 8:  # Start
                    print("Start Detected")
                    accept_action = True
                elif predicted_action_index == 7:  # Off
                    print("Off Detected")
                    accept_action = False


            # Xử lý cử chỉ tay trái (Zoom In, Zoom Out)
            if left_hand and accept_action == True:
                lm_list.append(make_landmark_timestep(left_hand))

                if predicted_action_index == 5:  # Zoom In
                    print("Zoom In Detected")
                elif predicted_action_index == 6:  # Zoom Out
                    print("Zoom Out Detected")

            # Xử lý cử chỉ tay phải (Move Mouse, Click)
            if right_hand and accept_action == True:
                lm_list.append(make_landmark_timestep(right_hand))

                xMouse = (int(right_hand.landmark[8].x * W)) / (W/1920)
                yMouse = (int(right_hand.landmark[8].y * H))/ (H/1080)

                if predicted_action_index == 0:  # Left Click
                    mouse.click(button='left')
                    print("Left Click Performed")
                elif predicted_action_index == 2:  # Right Click
                    mouse.click(button='right')
                    print("Right Click Performed")
                elif predicted_action_index == 1:  # Move Mouse
                    smooth_move(xMouse, yMouse, duration=0.1, steps=5)
                    print(predicted_action_index)

            # Vẽ tay
            if left_hand:
                mp_drawing.draw_landmarks(image, left_hand, mp_hands.HAND_CONNECTIONS)
                # Tay trai
                x__, y__ = get_hand_landmark(left_hand)
                x3 = int(min(x__) * W) - 10
                y3 = int(min(y__) * H) - 10

                x4 = int(max(x__) * W) - 10
                y4 = int(max(y__) * H) - 10

                cv2.rectangle(image, (x3, y3), (x4, y4), (0, 0, 0), 4)
                cv2.putText(image, predicted_action, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            if right_hand:
                mp_drawing.draw_landmarks(image, right_hand, mp_hands.HAND_CONNECTIONS)
                # Tay phai
                x_, y_ = get_hand_landmark(right_hand)
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(image, predicted_action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

    # Hiển thị hình ảnh
    cv2.imshow("Hand Tracking with Mouse Control", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
capture.release()
cv2.destroyAllWindows()

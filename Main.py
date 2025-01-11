# Import Libraries
import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf


# Lấy Mô hình Toàn diện từ Mediapipe và
# Khởi tạo Mô hình

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)

model = tf.keras.models.load_model("D:\All_learn_programs\Python\\virtualMouse\MODEL\model_rnn.h5")

# Khởi tạo các tiện ích vẽ để vẽ các điểm mốc trên khuôn mặt trên hình ảnh
mp_drawing = mp.solutions.drawing_utils
# (0) trong VideoCapture được sử dụng để kết nối với camera mặc định của máy tính của bạn
capture = cv2.VideoCapture(0)

# Khởi tạo thời gian hiện tại và thời gian quý báu để tính FPS
previousTime = 0
currentTime = 0

labels = {0:"Move Mouse", 1: "Left Mouse", 2:"RightMouse",3:"ScrollMouse"}
lm_list = []

def detect(model, lm_list):
    lm_list = np.expand_dims(np.array(lm_list), axis=0)
    results = model.predict(lm_list)
    return np.argmax(results, axis=1)

def make_landmark_timestep(results):
    return [lm.x for lm in results.right_hand_landmarks.landmark] + \
           [lm.y for lm in results.right_hand_landmarks.landmark] + \
           [lm.z for lm in results.right_hand_landmarks.landmark]
def get_hand_landmark(results):
    x_list = []
    y_list =[]
    for lm in results.right_hand_landmarks.landmark:
        x_list.append(lm.x)
        y_list.append(lm.y)
    return x_list,y_list

i = 0
warmup_frames = 60
sequence_length = 10
predicted_action = ""
while capture.isOpened():
    # chụp từng khung hình
    ret, frame = capture.read()
    # xoay camera nếu bị ngược
    frame = cv2.flip(frame, 1)
    # thay đổi kích thước khung hình để xem tốt hơn
    frame = cv2.resize(frame, (800, 600))
    H,W,_ = frame.shape

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

    i += 1
    x_ = []
    y_ = []

    if i > warmup_frames:
        if results.right_hand_landmarks:
            hand_landmarks = make_landmark_timestep(results)
            lm_list.append(hand_landmarks)

            if len(lm_list) > sequence_length:
                lm_list = lm_list[-sequence_length:]
            if len(lm_list) == sequence_length:
                prediction = detect(model, lm_list)
                predicted_action = labels.get(prediction[0])

                print("Action Id: ", prediction[0])
                print("Action Detected:", predicted_action)
                # cv2.putText(image, predicted_action, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                lm_list = []  # Reset list after prediction
            x_, y_ = get_hand_landmark(results)
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(image, predicted_action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        else:
            predicted_action = "No Action Detected"
            print("No right hand detected.")



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
        # print(landmark, landmark.value)
        pass

    # print(mp_holistic.HandLandmark.WRIST.value)


    # Nhập phím 'q' để phá vòng lặp
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Khi tất cả quá trình hoàn tất
# Giải phóng capture và hủy tất cả các cửa sổ
capture.release()
cv2.destroyAllWindows()


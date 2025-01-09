import csv
import time
import pandas as pd
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


def extract_hand_landmarks(results):
    landmarks = []
    for id, lm in enumerate(results.landmark):
        landmarks.append(lm.x)
        landmarks.append(lm.y)
        landmarks.append(lm.z)
    return landmarks


landmarks_data = []
# video = "Video_test/taytrai.mp4"
video = "Video_test/tayphai.mp4"
# Mở camera
capture = cv2.VideoCapture(video)


while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (800, 600))  # Resize khung hình
    if not ret:
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
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Lấy thông tin tay trái hoặc phải
            hand_label = results.multi_handedness[idx].classification[0].label

            # Chỉ xử lý tay phải
            if hand_label == "Right":
                h, w, c = image.shape
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # lấy tọa độ của ngón tay trên mô hình = indextip
                # print(index_tip)
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                # print(hand_landmarks.landmark)
                # xử lý data 1 khung hình thành 1 list gồm tọa độ 21 đểm
                hand_frame = extract_hand_landmarks(hand_landmarks)
                # print(hand_frame)
                # thêm vào data frame
                landmarks_data.append(hand_frame)
                # landmarks_data.append(hand_landmarks.landmark)
                # Vẽ điểm ngón chỏ trên khung hình
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(image, f"Right Hand({x},{y})", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Vẽ toàn bộ bàn tay
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Lưu vào df và xuất ra file csv
            columns = [str(i) for i in range(len(landmarks_data[0]))]
            df = pd.DataFrame(landmarks_data, columns=columns)
            df.to_csv("hand_landmarks.csv", index=False)
            print(f"lưu thành công {landmarks_data[0]}")
    # Hiển thị hình ảnh kết quả
    cv2.imshow("Hand Tracking with Mouse Control", image)


    # Nhập phím 'q' để phá vòng lặp
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


# Giải phóng tài nguyên
capture.release()
cv2.destroyAllWindows()

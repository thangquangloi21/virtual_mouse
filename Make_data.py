import os
import cv2
import pandas as pd
import mediapipe as mp

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_hand_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

# Thư mục chứa video và nơi lưu dữ liệu CSV
video_folder = r"D:\All_learn_programs\Python\virtualMouse\Video_test\BothHand"
output_folder = r"D:\All_learn_programs\Python\virtualMouse\Data\Both"

# Lấy danh sách tất cả các video trong thư mục
video_files = sorted(
    [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
)

# Xử lý từng video
for index, video_name in enumerate(video_files, start=1):
    video_path = os.path.join(video_folder, video_name)
    output_csv = os.path.join(output_folder, f"{index}.csv")

    print(f"Đang xử lý video: {video_name} ({index}/{len(video_files)})")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Không thể mở video {video_name}, bỏ qua.")
        continue

    landmarks_data = []

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print(f"Hoàn thành xử lý video: {video_name}")
            break

        # Resize khung hình
        try:
            frame = cv2.resize(frame, (800, 600))
        except cv2.error as e:
            print(f"Lỗi khi resize khung hình: {e}")
            continue

        # Chuyển đổi màu cho Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Dự đoán bàn tay
        results = hands_model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Nếu phát hiện bàn tay
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label

                # Xử lý tay phải hoặc tay trái
                h, w, _ = image.shape
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                # Trích xuất tọa độ bàn tay
                hand_frame = extract_hand_landmarks(hand_landmarks)
                landmarks_data.append(hand_frame)

                # Vẽ điểm ngón trỏ
                color = (0, 255, 0) if hand_label == "Right" else (255, 0, 0)
                cv2.circle(image, (x, y), 10, color, -1)
                cv2.putText(image, f"{hand_label} ({x},{y})", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Vẽ toàn bộ bàn tay
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Hiển thị kết quả
        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lưu dữ liệu landmarks vào file CSV
    if landmarks_data:
        columns = [str(i) for i in range(len(landmarks_data[0]))]
        df = pd.DataFrame(landmarks_data, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"Dữ liệu đã lưu: {output_csv}")

    # Giải phóng video
    capture.release()

# Đóng cửa sổ hiển thị
cv2.destroyAllWindows()
print("Hoàn tất xử lý tất cả video!")

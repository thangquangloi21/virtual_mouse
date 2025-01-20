import os
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Cấu hình tham số
no_of_timesteps = 10
num_of_epochs = 16
batch_size = 32

# Hàm tăng cường dữ liệu (Data Augmentation)
def augment_data(dataset):
    augmented_data = []
    for seq in dataset:
        # Thêm nhiễu Gaussian
        noise = np.random.normal(0, 0.01, seq.shape)
        seq_noisy = seq + noise
        # Dịch chuyển
        shift = np.random.uniform(-0.1, 0.1)
        seq_shifted = seq + shift
        # Thêm cả bản gốc và đã tăng cường
        augmented_data.extend([seq, seq_noisy, seq_shifted])
    return np.array(augmented_data)

# Hàm tạo Data Generator để tiết kiệm bộ nhớ
def data_generator(input_dir, label, batch_size, augment=False):
    X, y = [], []
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_csv = os.path.join(input_dir, filename)
            action_df = pd.read_csv(file_csv)
            dataset = action_df.iloc[:, 0:].values
            n_sample = len(dataset)
            for i in range(no_of_timesteps, n_sample):
                seq = dataset[i - no_of_timesteps:i, :]
                X.append(seq)
                y.append(label)
                if len(X) == batch_size:
                    X, y = np.array(X), np.array(y)
                    if augment:
                        X = augment_data(X)
                    yield X, y
                    X, y = [], []

# Đường dẫn dữ liệu
input_dirs = {
    "LeftMouse": 0,
    "MoveMouse": 1,
    "RightMouse": 2,
    "ScrollDown": 3,
    "ScrollUp": 4,
    "Start": 5,
    "PauseCursor": 6,
}
base_dir = "D:/All_learn_programs/Python/virtualMouse/Data"

# Đọc toàn bộ dữ liệu để tạo tập test
X, y = [], []
for action, label in input_dirs.items():
    for filename in os.listdir(os.path.join(base_dir, action)):
        if filename.endswith(".csv"):
            file_csv = os.path.join(base_dir, action, filename)
            action_df = pd.read_csv(file_csv)
            dataset = action_df.iloc[:, 0:].values
            n_sample = len(dataset)
            for i in range(no_of_timesteps, n_sample):
                X.append(dataset[i - no_of_timesteps:i, :])
                y.append(label)

X, y = np.array(X), np.array(y)
print(f"Dataset shape: {X.shape}, Labels: {y.shape}")

# Tách dữ liệu thành tập train/validation/test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Xây dựng mô hình GRU
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=32),
    Dropout(0.2),
    Dense(units=len(input_dirs), activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Callbacks để cải thiện quá trình huấn luyện
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)

# Huấn luyện mô hình
H = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=num_of_epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, lr_scheduler]
)

# Lưu mô hình
model.save("model_lstm.h5")

# Vẽ đồ thị kết quả
plt.plot(H.history['loss'], label='Training Loss')
plt.plot(H.history['val_loss'], label='Validation Loss')
plt.plot(H.history['accuracy'], label='Training Accuracy')
plt.plot(H.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()

# Đánh giá mô hình
model = load_model("model_lstm.h5")
loss, acc = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# Đánh giá F1 Score
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred, average='macro')
print("F1 Score:", f1)

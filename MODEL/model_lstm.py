import os
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model


# Cấu hình tham số
no_of_timesteps = 10
num_of_epochs = 16
batch_size = 32



# Đường dẫn dữ liệu
input_dirs = {
    "Move": 0,
    "Pause": 1,
    "Scroll": 2,
    "Start": 3,
}
base_dir = r"D:\All_learn_programs\Python\virtualMouse\Data"

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
    LSTM(units=40, return_sequences=True),
    Dropout(0.2),
    LSTM(units=30, return_sequences=True),
    Dropout(0.2),
    LSTM(units=20, return_sequences=True),
    Dropout(0.2),
    LSTM(units=10),
    Dropout(0.2),
    Dense(units=len(input_dirs), activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.build(input_shape=(None,X.shape[1], X.shape[2])) #This line builds the model, making it ready for summary and plotting

# Summary + Model Visualization
model.summary()
# plot_model(model, to_file="model_lstm.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)

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
model.save("model_lstm.keras")

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
model = load_model("model_lstm.keras")
loss, acc = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# Đánh giá F1 Score
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred, average='macro')
print("F1 Score:", f1)

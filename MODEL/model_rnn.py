import os
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import Sequential, load_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def read_data(inp_action, label):
    for filename in os.listdir(inp_action):
        if filename.endswith(".csv"):
            file_csv = os.path.join(inp_action, filename)
            action_df = pd.read_csv(file_csv)
            dataset = action_df.iloc[:, 0:].values
            n_sample = len(dataset)
            for i in range(no_of_timesteps, n_sample):
                X.append(dataset[i - no_of_timesteps:i, :])
                y.append(label)
            print(file_csv)


no_of_timesteps = 10
X = []
y = []
numOfEpoch = 16
# đọc dữ liệu
inp_action1 = "D:\All_learn_programs\Python\\virtualMouse\Data\LeftMouse"
inp_action2 = "D:\All_learn_programs\Python\\virtualMouse\Data\MoveMouse"
inp_action3 = "D:\All_learn_programs\Python\\virtualMouse\Data\RightMouse"
inp_action4 = "D:\All_learn_programs\Python\\virtualMouse\Data\ScrollDown"
inp_action5 = "D:\All_learn_programs\Python\\virtualMouse\Data\ScrollUp"
inp_action6 = "D:\All_learn_programs\Python\\virtualMouse\Data\ZoomIn"
inp_action7 = "D:\All_learn_programs\Python\\virtualMouse\Data\Zoomout"
inp_action8 = "D:\All_learn_programs\Python\\virtualMouse\Data\Off"
inp_action9 = "D:\All_learn_programs\Python\\virtualMouse\Data\Start"
inp_action10 = "D:\All_learn_programs\Python\\virtualMouse\Data\PauseCursor"

read_data(inp_action1, 0)
read_data(inp_action2, 1)
read_data(inp_action3, 2)
read_data(inp_action4, 3)
read_data(inp_action5, 4)
read_data(inp_action6, 5)
read_data(inp_action7, 6)
read_data(inp_action8, 7)
read_data(inp_action9, 8)
read_data(inp_action10, 9)
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
X_train, x_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)

# Xây dựng mô hình
model = Sequential()
model.add(SimpleRNN(units=40, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=40, return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=40, return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=40))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation="softmax"))
model.compile(optimizer="adam", metrics=['accuracy'], loss="sparse_categorical_crossentropy")

# Huấn luyện mô hình
H = model.fit(X_train, y_train, epochs=numOfEpoch, batch_size=32, validation_data=(x_val, y_val))
model.save("model_rnn.h5")


plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='Training Loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='Validation Loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='Training Accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()

model = load_model("model_rnn.h5")
loss, acc = model.evaluate(X_test, y_test)
print("loss:", loss)
print("acc:", acc)

# Đánh giá mô hình f1 scope
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred, average='macro')

print("F1 score:", f1)

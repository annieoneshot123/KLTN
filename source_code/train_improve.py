import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint

# Read folder names and labels from the CSV file
csv_file = "/home/hai20521281/Downloads/TEST/labels_improve.csv"
dataset_path = "/home/hai20521281/Downloads/TEST/dataset_improve"
folder_names = []
labels = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        folder_names.append(row[0])
        labels.append(int(row[1]))

# Prepare the input data
def preprocess(vector1, vector2):
    res = []
    res.append(eval(vector1))
    res.append(eval(vector2))
    return res

def get_file_in_folder(folder_path):
    tmp = os.listdir(folder_path)
    if len(tmp) == 1:
        return tmp[0], tmp[0]
    else:
        file1 = tmp[0]
        file2 = tmp[1]
        file_name = os.path.basename(folder_path)
        if file2[:5] == file_name[:5]:
            temp = file2
            file2 = file1
            file1 = temp
        return file1, file2

from tqdm import tqdm
X = []
for folder_name in tqdm(folder_names):
    folder_path = os.path.join(dataset_path, folder_name)
    file1, file2 = get_file_in_folder(folder_path)
    file1_path = os.path.join(folder_path, file1)
    file2_path = os.path.join(folder_path, file2)
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        vector1 = file1.read().strip()
        vector2 = file2.read().strip()
    feature_vector = preprocess(vector1, vector2)
    X.append(feature_vector)

# Convert the input data to NumPy array
X = np.array(X)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train)

# Define callback to save model
checkpoint_path = "/home/hai20521281/Downloads/TEST/checkpoints_improve/model_{epoch:02d}.h5"
model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)

def CNN_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((2, 1024, 1), input_shape=(2, 1024)))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(2, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(8, (1, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def LSTM_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape=(2, 1024)))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def CNN_LSTM_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((2, 1024, 1), input_shape=(2, 1024)))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(2, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(8, (1, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.LSTM(50))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def CNN_GRU_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((2, 1024, 1), input_shape=(2, 1024)))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(2, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(8, (1, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.GRU(3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# CNN
model_cnn = CNN_model()
model_cnn.summary()
model_cnn.fit(X_train, to_categorical(y_train, 2), epochs=3, batch_size=32, validation_data=(X_test, to_categorical(y_test, 2)),
              callbacks=[model_checkpoint])

# LSTM
model_lstm = LSTM_model()
model_lstm.summary()
model_lstm.fit(X_train, to_categorical(y_train, 2), epochs=3, batch_size=32, validation_data=(X_test, to_categorical(y_test, 2)),
               callbacks=[model_checkpoint])

# CNN + LSTM
model_cnn_lstm = CNN_LSTM_model()
model_cnn_lstm.summary()
model_cnn_lstm.fit(X_train, to_categorical(y_train, 2), epochs=3, batch_size=32, validation_data=(X_test, to_categorical(y_test, 2)),
                   callbacks=[model_checkpoint])

# CNN + GRU
model_cnn_gru = CNN_GRU_model()
model_cnn_gru.summary()
model_cnn_gru.fit(X_train, to_categorical(y_train, 2), epochs=3, batch_size=32, validation_data=(X_test, to_categorical(y_test, 2)),
                  callbacks=[model_checkpoint])

# Save models after training
model_cnn.save('/home/hai20521281/Downloads/TEST/models/model_cnn_improve.h5')
model_lstm.save('/home/hai20521281/Downloads/TEST/models/model_lstm_improve.h5')
model_cnn_lstm.save('/home/hai20521281/Downloads/TEST/models/model_cnn_lstm_improve.h5')
model_cnn_gru.save('/home/hai20521281/Downloads/TEST/models/model_cnn_gru_improve.h5')

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    precision = precision_score(y_test_labels, y_pred_labels)
    recall = recall_score(y_test_labels, y_pred_labels)
    f1 = f1_score(y_test_labels, y_pred_labels)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    return precision, recall, f1, accuracy

# Evaluate CNN
precision_cnn, recall_cnn, f1_cnn, accuracy_cnn = evaluate_model(model_cnn, X_test, to_categorical(y_test, 2))
print(f"CNN Improve Model: Precision: {precision_cnn}, Recall: {recall_cnn}, F1 Score: {f1_cnn}, Accuracy: {accuracy_cnn}")

# Evaluate LSTM
precision_lstm, recall_lstm, f1_lstm, accuracy_lstm = evaluate_model(model_lstm, X_test, to_categorical(y_test, 2))
print(f"LSTM Improve Model: Precision: {precision_lstm}, Recall: {recall_lstm}, F1 Score: {f1_lstm}, Accuracy: {accuracy_lstm}")

# Evaluate CNN + LSTM
precision_cnn_lstm, recall_cnn_lstm, f1_cnn_lstm, accuracy_cnn_lstm = evaluate_model(model_cnn_lstm, X_test, to_categorical(y_test, 2))
print(f"CNN + LSTM Improve Model: Precision: {precision_cnn_lstm}, Recall: {recall_cnn_lstm}, F1 Score: {f1_cnn_lstm}, Accuracy: {accuracy_cnn_lstm}")

# Evaluate CNN + GRU
precision_cnn_gru, recall_cnn_gru, f1_cnn_gru, accuracy_cnn_gru = evaluate_model(model_cnn_gru, X_test, to_categorical(y_test, 2))
print(f"CNN + GRU Improve Model: Precision: {precision_cnn_gru}, Recall: {recall_cnn_gru}, F1 Score: {f1_cnn_gru}, Accuracy: {accuracy_cnn_gru}")

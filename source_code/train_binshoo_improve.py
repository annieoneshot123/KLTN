import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Dropout, TimeDistributed, Flatten, Dense, concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D, Reshape

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

def Siamese_Neural_Network(input_shape=(2, 1024), lstm_units=100, cnn_filters=64, dense_units=64):
    input_layer = Input(shape=input_shape)

    # Reshape the input to (time_steps, features) for Conv2D
    reshaped_input = Reshape((input_shape[0], input_shape[1], 1))(input_layer)

    # CNN Branch
    cnn_branch = Conv2D(filters=cnn_filters, kernel_size=(3, 1), activation='relu', padding='same')(reshaped_input)
    cnn_branch = MaxPooling2D(pool_size=(2, 1), padding='same')(cnn_branch)

    # Add more Conv2D and MaxPooling2D layers for deeper representation
    cnn_branch = Conv2D(filters=128, kernel_size=(3, 1), activation='relu', padding='same')(cnn_branch)
    cnn_branch = MaxPooling2D(pool_size=(2, 1), padding='same')(cnn_branch)

    # Flatten the spatial dimensions
    cnn_branch = Flatten()(cnn_branch)

    # LSTM Branch 1
    rnn_branch_1 = LSTM(lstm_units, return_sequences=True)(input_layer)
    rnn_branch_1 = LSTM(lstm_units)(rnn_branch_1)

    # LSTM Branch 2 (identical to Branch 1)
    rnn_branch_2 = LSTM(lstm_units, return_sequences=True)(input_layer)
    rnn_branch_2 = LSTM(lstm_units)(rnn_branch_2)

    # Concatenate branches
    merged = concatenate([cnn_branch, rnn_branch_1, rnn_branch_2], axis=-1)

    # Add Dense layers for deeper representation learning
    merged = Dense(256, activation='relu')(merged)
    merged = Dense(128, activation='relu')(merged)

    # Softmax classifier
    output_layer = Dense(2, activation='softmax')(merged)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Siamese Neural Network
model_siamese = Siamese_Neural_Network(input_shape=(2, 1024), lstm_units=100, dense_units=64)
model_siamese.summary()
model_siamese.fit(X_train, to_categorical(y_train, 2), epochs=3, batch_size=32, validation_data=(X_test, to_categorical(y_test, 2)),
                             callbacks=[model_checkpoint])

# Save model after training
model_siamese.save('/home/hai20521281/Downloads/TEST/models/binshoo_improve.h5')

# Evaluate Siamese Neural Network
def evaluate_model_siamese(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    precision = precision_score(y_test_labels, y_pred_labels)
    recall = recall_score(y_test_labels, y_pred_labels)
    f1 = f1_score(y_test_labels, y_pred_labels)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    return precision, recall, f1, accuracy

precision_siamese, recall_siamese, f1_siamese, accuracy_siamese = evaluate_model_siamese(model_siamese, X_test, to_categorical(y_test, 2))
print(f"BINSHOO: Precision: {precision_siamese}, Recall: {recall_siamese}, F1 Score: {f1_siamese}, Accuracy: {accuracy_siamese}")

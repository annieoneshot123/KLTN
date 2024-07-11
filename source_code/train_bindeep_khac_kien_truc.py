import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Read folder names and labels from the CSV file
csv_file = "/home/hai20521281/Downloads/bindeep/labels_khac_kien_truc.csv"
dataset_path = "/home/hai20521281/Downloads/bindeep/Khác kiến trúc, khác tối ưu hóa và khác trình biên dịch"
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
for folder_name in folder_names:
    folder_path = os.path.join(dataset_path, folder_name)
    file1, file2 = get_file_in_folder(folder_path)
    file1_path = os.path.join(folder_path, file1)
    file2_path = os.path.join(folder_path, file2)
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        vector1 = file1.read().strip()
        vector2 = file2.read().strip()

    feature_vector = preprocess(vector1, vector2)
    X.append(feature_vector)

X = np.array(X)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture with different layers
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(2, 1024), return_sequences=True),  # LSTM layer with more units
    tf.keras.layers.Conv1D(512, 1, activation='relu'),  # Convolutional layer with more filters and ReLU activation
    tf.keras.layers.Conv1D(256, 1, activation='relu'),  # Additional convolutional layer
    tf.keras.layers.GlobalMaxPooling1D(),  # Global max pooling
    tf.keras.layers.Dense(2, activation='softmax')  # Output layer
])

# Compile the model with a different optimizer and compiler
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Fit the model
batch_size = 32
epochs = 3
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))



# Predict labels for test data
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate precision, recall, F1 score, and accuracy
precision = precision_score(y_test, y_pred_labels)
recall = recall_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)
accuracy = accuracy_score(y_test, y_pred_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)

# Save the model
model.save('model_bindeep_khac_kien_truc.h5')

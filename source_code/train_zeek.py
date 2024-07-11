import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

def preprocess(vector1,vector2):
    res=[]
    res.append(eval(vector1))
    #print(vector2)
    res.append(eval(vector2))
    return res

def get_file_in_folder(folder_path):
    tmp=os.listdir(folder_path)
    if len(tmp)==1:
        return tmp[0],tmp[0]
    else:
        file1=tmp[0]
        file2=tmp[1]
        file_name=os.path.basename(folder_path)
        if file2[:5]==file_name[:5]:
            temp=file2
            file2=file1
            file1=temp
        return file1,file2



from tqdm import tqdm
X = []
for folder_name in tqdm(folder_names):
    folder_path = os.path.join(dataset_path, folder_name)
    file1,file2 = get_file_in_folder(folder_path)
    file1_path = os.path.join(folder_path, file1)
    file2_path = os.path.join(folder_path, file2)
    #print(file2_path)
    # Read the content of the text files
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        vector1 = file1.read().strip()
        vector2 = file2.read().strip()

    # Preprocess the string vectors and combine them into one feature vector
    # Here you need to implement the preprocessing steps suitable for your model
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



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(256, 1, activation='tanh', input_shape=(2, 1024)),  # Convolutional layer with 512 filters and kernel size of 3
    tf.keras.layers.Conv1D(128, 1, activation='tanh'),  # Convolutional layer with 256 filters and kernel size of 3
    tf.keras.layers.Flatten(),  # Flatten the output for dense layers
    tf.keras.layers.Dense(2, activation='softmax')  # Output layer with 2 units and softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Add dropout regularization
dropout_rate = 0.1
model.add(tf.keras.layers.Dropout(dropout_rate))

# Print the model summary
model.summary()

# Fit the model
batch_size = 32
epochs = 3
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.utils import to_categorical
y_train_encoded = to_categorical(y_train, 2)
y_test_encoded = to_categorical(y_test, 2)
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert one-hot encoded labels back to single-digit labels
y_test_labels = np.argmax(y_test_encoded, axis=1)

# Calculate the scores
precision = precision_score(y_test_labels, y_pred_labels)
recall = recall_score(y_test_labels, y_pred_labels)
f1 = f1_score(y_test_labels, y_pred_labels)
accuracy = accuracy_score(y_test_labels, y_pred_labels)
model.save('model_zeek.h5')
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)

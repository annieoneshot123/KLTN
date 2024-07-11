import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

# Define a function to get files in a folder
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


# Read folder names and labels from the CSV file
csv_file = "/home/hai20521281/Downloads/TEST/labels_eval.csv"
dataset_path = "/home/hai20521281/Downloads/TEST/dataset_eval"
folder_names = []
labels = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        folder_names.append(row[0])
        labels.append(int(row[1]))

# Prepare the input data
X = []
for folder_name in tqdm(folder_names):
    folder_path = os.path.join(dataset_path, folder_name)
    file1, file2 = get_file_in_folder(folder_path)
    file1_path = os.path.join(folder_path, file1)
    file2_path = os.path.join(folder_path, file2)
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        vector1 = file1.read().strip()
        vector2 = file2.read().strip()
    # Combine vector1 and vector2 as input text
    input_text = vector1 + " " + vector2
    X.append(input_text)

# Convert the input data to NumPy array
X = np.array(X)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode the input data
X_train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, return_tensors='tf', max_length=512)
X_test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, return_tensors='tf', max_length=512)

# Create BERT-based model
input_ids = Input(shape=(512,), dtype=tf.int32)
outputs = bert_model(input_ids)[1]  # Get the pooled output
output_layer = Dense(2, activation='softmax')(outputs)
model = Model(inputs=input_ids, outputs=output_layer)

# Compile the model
optimizer = Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Fit the model
batch_size = 8
epochs = 3
model.fit(X_train_encodings['input_ids'], y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_encodings['input_ids'], y_test))

# Evaluate the model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

y_pred = model.predict(X_test_encodings['input_ids'])
y_pred_labels = np.argmax(y_pred, axis=1)

precision = precision_score(y_test, y_pred_labels)
recall = recall_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)
accuracy = accuracy_score(y_test, y_pred_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)

# Save the model
model.save('model_BERT.h5')


import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from keras.utils import to_categorical
import pandas as pd

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

# Load models from the 'modules' folder
model_folder = 'models'
model_files = os.listdir(model_folder)
models = [tf.keras.models.load_model(os.path.join(model_folder, model_file)) for model_file in model_files]

results = []

for model, model_file in zip(models, model_files):
    y_encoded = to_categorical(y, 2)
    y_pred = model.predict(X)
    y_pred_labels = np.argmax(y_pred, axis=1)

    y_labels = np.argmax(y_encoded, axis=1)

    precision = precision_score(y_labels, y_pred_labels)
    recall = recall_score(y_labels, y_pred_labels)
    f1 = f1_score(y_labels, y_pred_labels)
    accuracy = accuracy_score(y_labels, y_pred_labels)

    cm = confusion_matrix(y_labels, y_pred_labels)

    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    results.append({
        'Model': model_file,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'True Positive (TP)': TP,
        'False Positive (FP)': FP,
        'False Negative (FN)': FN,
        'True Negative (TN)': TN
    })

# Hiển thị kết quả bằng pandas DataFrame
results_df = pd.DataFrame(results)
print(results_df)


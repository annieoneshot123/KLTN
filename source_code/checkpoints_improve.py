import os
import csv
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

def load_data(csv_file, dataset_path):
    folder_names = []
    labels = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            folder_names.append(row[0])
            labels.append(int(row[1]))

    X = []
    y = []

    for folder_name, label in zip(folder_names, labels):
        folder_path = os.path.join(dataset_path, folder_name)
        file1, file2 = get_file_in_folder(folder_path)
        file1_path = os.path.join(folder_path, file1)
        file2_path = os.path.join(folder_path, file2)

        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            vector1 = file1.read().strip()
            vector2 = file2.read().strip()

        feature_vector = preprocess(vector1, vector2)
        X.append(feature_vector)
        y.append(label)

    return np.array(X), np.array(y)

def load_models_from_checkpoints(checkpoints_folder):
    models = []
    for file_name in os.listdir(checkpoints_folder):
        if file_name.endswith(".h5"):
            model_path = os.path.join(checkpoints_folder, file_name)
            model = load_model(model_path)
            models.append(model)
    return models

def evaluate_models(models, X_test, y_test):
    results = []
    for model in models:
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = y_test
        precision = precision_score(y_test_labels, y_pred_labels)
        recall = recall_score(y_test_labels, y_pred_labels)
        f1 = f1_score(y_test_labels, y_pred_labels)
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        results.append((precision, recall, f1, accuracy))
    return results

def main():
    csv_file = "/home/hai20521281/Downloads/TEST/labels_improve.csv"
    dataset_path = "/home/hai20521281/Downloads/TEST/dataset_improve"

    X_test, y_test = load_data(csv_file, dataset_path)

    checkpoints_folder = "/home/hai20521281/Downloads/TEST/checkpoints_improve"
    models = load_models_from_checkpoints(checkpoints_folder)

    for epoch, model in enumerate(models, start=1):
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = y_test
        precision = precision_score(y_test_labels, y_pred_labels)
        recall = recall_score(y_test_labels, y_pred_labels)
        f1 = f1_score(y_test_labels, y_pred_labels)
        accuracy = accuracy_score(y_test_labels, y_pred_labels)

        print(f"Epoch {epoch} - Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}")

if __name__ == "__main__":
    main()

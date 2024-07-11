import os
import csv
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

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
sentences = []
for folder_name in folder_names:
    folder_path = os.path.join(dataset_path, folder_name)
    file1, file2 = get_file_in_folder(folder_path)
    file1_path = os.path.join(folder_path, file1)
    file2_path = os.path.join(folder_path, file2)
    # Read the content of the text files
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        vector1 = file1.read().strip()
        vector2 = file2.read().strip()
        sentences.append(vector1.split())  # Split text into words and append to sentences
        sentences.append(vector2.split())

# Train the Word2Vec model with skip-gram
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Save the Word2Vec model
word2vec_model.save("word2vec_model.bin")

# Now you can use the word2vec_model for your subsequent tasks


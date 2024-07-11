import os
import shutil
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def load_data_from_folder(folder_path):
    """
    Load dữ liệu từ thư mục và trả về danh sách các vector.
    """
    data = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r') as file:
            content = file.read()
            # Tách dữ liệu thành danh sách các số nguyên
            numbers = [int(x.strip('[],')) for x in content.split(',') if x.strip().isdigit()]
            if len(numbers) >= 3:  # Chỉ thêm vector có đủ 3 giá trị vào danh sách
                data.append(numbers)
    return data

def entropy(labels):
    """
    Tính entropy của vector 0-1.
    """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0

    return -np.sum(probs * np.log(probs)) / np.log(n_classes)

def determine_environment(vector1, vector2):
    entropy1 = entropy(vector1)
    entropy2 = entropy(vector2)

    # Đặt ngưỡng cho sự khác biệt entropy
    threshold = 0.01

    if abs(entropy1 - entropy2) < threshold:
        return "Cùng một kiến trúc, khác tối ưu hóa và khác trình biên dịch"
    elif entropy1 > entropy2:
        return "Cùng một tối ưu hóa, khác kiến trúc và khác trình biên dịch"
    else:
        return "Khác kiến trúc, khác tối ưu hóa và khác trình biên dịch"

# Đường dẫn tới thư mục chứa dữ liệu
root_folder_path = "/home/hai20521281/Downloads/bindeep/dataset_bindeep"

# Thư mục kết quả
results_folders = [
    "Cùng một kiến trúc, khác tối ưu hóa và khác trình biên dịch",
    "Cùng một tối ưu hóa, khác kiến trúc và khác trình biên dịch",
    "Khác kiến trúc, khác tối ưu hóa và khác trình biên dịch"
]

# Tạo thư mục kết quả
for result_folder in results_folders:
    os.makedirs(os.path.join("/home/hai20521281/Downloads/bindeep", result_folder), exist_ok=True)

# Lặp qua tất cả các thư mục con trong thư mục gốc
for subdir in os.listdir(root_folder_path):
    subfolder_path = os.path.join(root_folder_path, subdir)
    if os.path.isdir(subfolder_path):
        print(f"\nAnalyzing samples in folder: {subdir}")
        data = load_data_from_folder(subfolder_path)

        # Chọn các cặp dữ liệu và phân loại môi trường
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                vector1 = data[i]
                vector2 = data[j]
                ket_qua = determine_environment(vector1, vector2)
                print(f"Môi trường: {ket_qua}")

                # Tạo thư mục đích dựa trên kết quả phân loại
                destination_folder = os.path.join("/home/hai20521281/Downloads/bindeep", ket_qua)

                # Sao chép thư mục chứa file1.txt và file2.txt vào thư mục đích
                shutil.copytree(subfolder_path, os.path.join(destination_folder, subdir))


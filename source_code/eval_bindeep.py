import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

model = tf.keras.models.load_model('model_bindeep_khac_kien_truc.h5')

def cal_similarity_score(function1, function2):
    # Khởi tạo một mảng zeros với kích thước mong muốn (1, 2, 1024)
    target = np.zeros((1, 2, 1024))
    
    # Chèn dữ liệu hiện có vào mảng zeros, để phần còn lại là padding
    target[0, 0, :512] = function1  # Chèn function1 vào nửa đầu của chiều thứ 2
    target[0, 1, :512] = function2  # Chèn function2 vào nửa đầu của chiều thứ 2
    
    # Sử dụng mô hình để dự đoán
    res = model.predict(target)
    return res[0][1]

def load_data_to_eval(link):
    list_file = os.listdir(link)
    print(len(list_file))
    print(list_file[0])
    data = []
    for file in tqdm(list_file):
        with open(os.path.join(link, file), 'r') as content_file:
            content = content_file.read().strip()
            # Giả sử 'content' chứa dữ liệu bạn cần dưới dạng thích hợp
            data.append(eval(content))
    return data

data = load_data_to_eval('/home/hai20521281/Downloads/Dataset/eval/eval_O1_Os_poolsize10/')

reciprocal_ranks = []
recall_at_1 = 0
for sample in tqdm(data):
    target_func = sample[0]
    source_func = sample[1]
    function_pool = sample[2]
    similarity_scores = []
    for function in function_pool:
        similarity_score = cal_similarity_score(source_func, function)
        similarity_scores.append(similarity_score)

    sorted_functions = [f for _, f in sorted(zip(similarity_scores, function_pool), reverse=True)]

    rank = sorted_functions.index(target_func) + 1
    print(f"Rank: {rank}")
    reciprocal_rank = 1.0 / rank
    reciprocal_ranks.append(reciprocal_rank)
    if rank == 1:
        recall_at_1 += 1

mrr = np.mean(reciprocal_ranks)
recall_at_1 = recall_at_1 / len(data)
print("Recall@1 Score:", recall_at_1)
print("MRR Score:", mrr)

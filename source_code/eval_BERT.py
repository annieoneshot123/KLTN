import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
from transformers import TFBertModel

# Đăng ký lớp TFBertModel như là một đối tượng tùy chỉnh
tf.keras.utils.get_custom_objects().update({'TFBertModel': TFBertModel})

def tai_mo_hinh(duong_dan):
    return load_model(duong_dan)

def tinh_diem_tuong_dong(ham1, ham2, model):
    # Tạo một mảng zeros
    target = np.zeros((1, 2, 1024))
    target[0, 0, :512] = ham1
    target[0, 1, :512] = ham2

    # Cắt mảng `target` thành hai mảng con, mỗi mảng có 512 phần tử
    target1 = target[:, 0, :512]
    target2 = target[:, 1, :512]

    # Dự đoán sử dụng mô hình
    ket_qua1 = model.predict(target1)
    ket_qua2 = model.predict(target2)

    # Tính điểm tương đồng
    diem = np.sum(ket_qua1 * ket_qua2) / np.linalg.norm(ket_qua1) / np.linalg.norm(ket_qua2)

    return diem

def tai_du_lieu(link):
    danh_sach_file = os.listdir(link)
    print(len(danh_sach_file))
    print(danh_sach_file[0])
    du_lieu = []
    for file in tqdm(danh_sach_file):
        with open(os.path.join(link, file), 'r') as file_noi_dung:
            noi_dung = file_noi_dung.read().strip()
            du_lieu.append(eval(noi_dung))
    return du_lieu

# Tải mô hình
model = tai_mo_hinh('BERT.h5')

# Tải và xử lý dữ liệu đánh giá
du_lieu = tai_du_lieu('/home/hai20521281/Downloads/Dataset/eval/eval_O1_Os_poolsize10/')

hang_dao_nguoc = []
recall_at_1 = 0
for mau in tqdm(du_lieu):
    ham_muc_tieu = mau[0]
    ham_nguon = mau[1]
    bo_ham = mau[2]
    diem_tuong_dong = []
    for ham in bo_ham:
        diem = tinh_diem_tuong_dong(ham_nguon, ham, model)
        diem_tuong_dong.append(diem)

    ham_sap_xep = [f for _, f in sorted(zip(diem_tuong_dong, bo_ham), reverse=True)]

    hang = ham_sap_xep.index(ham_muc_tieu) + 1
    hang_dao_nguoc.append(1.0 / hang)
    if hang == 1:
        recall_at_1 += 1

mrr = np.mean(hang_dao_nguoc)
recall_at_1 = recall_at_1 / len(du_lieu)
print("Điểm Recall@1:", recall_at_1)
print("Điểm MRR:", mrr)


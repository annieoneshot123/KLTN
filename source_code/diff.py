import pyvex
import archinfo
import os 
import random 
import pickle
from playdata import *
from analyse import *
from gensim.models import Word2Vec
import numpy as np
import gensim
from collections import defaultdict
import shutil

def get_vexIR(raw_bytes):
    irsb = pyvex.IRSB(data=raw_bytes, mem_addr=0x08048429, arch=archinfo.ArchX86())
    result=b''
    for stmt in irsb.statements:
        if not isinstance(stmt, pyvex.IRStmt.IMark):
            result+=(str(stmt).encode()+b'\n')
    return result.decode()

def process_a_func(raw_bytes, size, word2vec_model):
    vex_ir = get_vexIR(raw_bytes)
    block = vex_ir.split("\n")[:-1]
    strand = parse_bb_to_strand(block)
    
    # Chạy các phần tử trong strand qua Word2Vec để nhúng thành vector
    embedded_strand = []
    for i in strand:
        embedded_i = []
        for token in i:
            if token in word2vec_model.wv:
                embedded_i.append(word2vec_model.wv[token])
            else:
                # Nếu token không tồn tại trong từ điển của Word2Vec, thêm vector 0
                embedded_i.append([0] * word2vec_model.vector_size)
        embedded_strand.append(embedded_i)
    
    # Tính toán vector đại diện cho strand
    vector = extract_vector_from_strand(embedded_strand, size)
    
    return vector

# Trước khi gọi hàm process_a_func, cần load mô hình Word2Vec
word2vec_model = Word2Vec.load("/home/hai20521281/Downloads/bindeep/word2vec_model.bin")

def parse_full_filename_to_file_name(filename):
    return "-".join(filename.split("/")[-1].split('-')[:-1])


def save_res_diff(vector1, vector2, result_folder, filename1, filename2, func_name1, func_name2):
    # Tạo tên thư mục dựa trên tên của hai tệp và hai hàm
    folder_name = f"{parse_full_filename_to_file_name(filename1)}--{func_name1}___{parse_full_filename_to_file_name(filename2)}--{func_name2}"
    
    # Giới hạn chiều dài tên thư mục
    max_folder_name_length = 200  # Đặt giới hạn tùy thuộc vào yêu cầu của hệ thống hoặc phong cách lập trình
    
    if len(folder_name) > max_folder_name_length:
        folder_name = folder_name[:max_folder_name_length]  # Cắt tên thư mục nếu quá dài
    
    new_folder = os.path.join(result_folder, folder_name)
    os.mkdir(new_folder)

    # Lưu vector1 vào tệp
    with open(os.path.join(new_folder, f"{parse_full_filename_to_file_name(filename1)}--{func_name1}.txt"), 'wb') as w1:
        w1.write(str(vector1).encode())

    # Lưu vector2 vào tệp
    with open(os.path.join(new_folder, f"{parse_full_filename_to_file_name(filename2)}--{func_name2}.txt"), 'wb') as w2:
        w2.write(str(vector2).encode()) 


dif_result_path='/home/hai20521281/Downloads/bindeep/diff_bindeep/'

sample_dir='/home/hai20521281/Downloads/TEST/sample/'
list_folder=os.listdir(sample_dir)

arr_file_name=[]
for folder in list_folder:
    list_filename=os.listdir(sample_dir+folder)
    for file_name in list_filename:
        if file_name=="saved_index.pkl":
            pass
        else:
            arr_file_name.append(sample_dir+folder+"/"+file_name)

count_target=2390
count_result=0
dem = 0
while True:
    if count_result == count_target:
        break
    for file_name in arr_file_name:
        data_pkl=pickle.load(open(file_name,'rb'))
        tmp_arr=arr_file_name
        tmp_arr.remove(file_name)
        functions=data_pkl.keys()
        #print(functions)
        for function in functions:
            try:
        
                current_vector=process_a_func(data_pkl[function][-3],1024, word2vec_model)
                count=0
                if count_result == count_target:
                            print("Xong roi neeeee")	
                            break
                while True:
                    if count==4:
                        break
                    another_file=random.choice(tmp_arr)
                    if os.path.basename(another_file)[:5]==os.path.basename(file_name)[:5]:
                        continue
                    else:
                        another_file_pkl=pickle.load(open(another_file,'rb'))
                        another_functions=list(another_file_pkl.keys())
                        another_function=random.choice(another_functions)
                        another_vector=process_a_func(another_file_pkl[another_function][-3],1024, word2vec_model)
                        #print(another_file_pkl[another_function][-3])
                        save_res_diff(current_vector, another_vector, dif_result_path, file_name, another_file, function, another_function)
                        #print("##############################")
                        #print(f"Currend FUNCTION: {function}")
                        #print(f"filename 1: {file_name}")
                        #print(f"filename 2: {another_file}")
                        #print(f"function 1: {function}")
                        #print(f"function 2: {another_function}")
                        #print("##############################")
                        count+=1
                        #print(f"count: {count}")
                        count_result+=1
                        print(f"{count_result}")
                        if count_result == count_target:
                            print("Sap xong oiii")
                            break
                            
                	
            except Exception as ex:
                print(ex)
                continue

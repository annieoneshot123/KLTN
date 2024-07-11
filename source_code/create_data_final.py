import pyvex
import archinfo
import os 
import random 
import pickle
from playdata import *
from analyse import *

def get_vexIR(raw_bytes):
    irsb = pyvex.IRSB(data=raw_bytes, mem_addr=0x08048429, arch=archinfo.ArchX86())
    result = b''
    for stmt in irsb.statements:
        if not isinstance(stmt, pyvex.IRStmt.IMark):
            result += (str(stmt).encode() + b'\n')
    return result.decode()
    
def process_a_func(raw_bytes, size):
    vex_ir = get_vexIR(raw_bytes)
    block = vex_ir.split("\n")[:-1]
    strand = parse_bb_to_strand(block)
    res = [copy_propagation(i) for i in strand]
    res = [dead_code_elimination(i) for i in res]
    res = [normalize(i) for i in res]
    vector = extract_vector_from_strand(strand)
    return vector

sample_dir = '/home/hai20521281/Downloads/Dataset/sample/'
list_folder = os.listdir(sample_dir)

arr_file_name = []
for folder in list_folder:
    list_filename = os.listdir(sample_dir + folder)
    for file_name in list_filename:
        if file_name != "saved_index.pkl":
            arr_file_name.append(sample_dir + folder + "/" + file_name)

def get_random_function(poolsize, func_name):
    try:
        file_to_load = random.choice(arr_file_name)
        data_pkl = pickle.load(open(file_to_load, 'rb'))
        functions = data_pkl.keys()
        choose_funcs = random.choices(list(functions), k=poolsize)
        if all(x != func_name for x in choose_funcs):
            result = []
            for choose_func in choose_funcs:
                tmp = data_pkl[choose_func]
                result.append(process_a_func(tmp[-3], 512))
            return result
        else:
            return []
    except: 
        return []

def create_result(res_arr, result_folder, res_num, func_name, pool_size):
    num = res_num
    for idx1 in range(len(res_arr)):
        for idx2 in range(len(res_arr)):
            arr_res = []
            arr_res.append(res_arr[idx1])
            arr_res.append(res_arr[idx2])
            pool = []
            pool.append(arr_res[0])
            while True:
                app_pool = get_random_function(pool_size-1, func_name)
                if len(app_pool) == pool_size-1:
                    break
            pool += app_pool
            arr_res.append(pool)
            with open(result_path+'sample'+str(num)+'.txt', 'wb') as w:
                w.write(str(arr_res).encode())
            num += 1    
            print(f"Created similarity pari function: {num}")

options = ['O2', 'O3']
pool_size = 10
result_path = '/home/hai20521281/Downloads/Dataset/eval/eval_O2_O3_poolsize30/'
dataset = DatasetBase('/home/hai20521281/Downloads/Dataset/sample/', None, True, options)
dataset.load_pair_data()

cnt = 0
ft_dataset = dataset.get_paired_data()
print("loaded")
ff = {}
for proj, func_name, func_data in ft_dataset:
    res = []
    try: 
        for opt in options:
            func_addr, asm_list, rawbytes_list, cfg, biai_featrue = func_data[opt]
            res.append(process_a_func(rawbytes_list, 1024))
        create_result(res, result_path, cnt, func_name, pool_size)
        cnt += 4
        if proj not in ff:
            ff[proj] = 1
        else:
            ff[proj] += 1
    except Exception as ex:
        print(ex)
    print("##############################")
print(cnt)
print(ff)


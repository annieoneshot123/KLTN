import pyvex
import archinfo
import os 
import random 
import pickle
from playdata import *
from analyse import *


def get_vexIR(raw_bytes):
    irsb = pyvex.IRSB(data=raw_bytes, mem_addr=0x08048429, arch=archinfo.ArchX86())
    result=b''
    for stmt in irsb.statements:
        if not isinstance(stmt, pyvex.IRStmt.IMark):
            result+=(str(stmt).encode()+b'\n')
    return result.decode()
    
def process_a_func(raw_bytes,size):
    vex_ir=get_vexIR(raw_bytes)
    #print(f"vex_ir : {vex_ir}")
    #print(type(vex_ir))
    #print("############")
    block=vex_ir.split("\n")[:-1]
    #print(f"split : {block}")
    #print(type(block))
    strand=parse_bb_to_strand(block)
    #print(f"strand : {strand}")
    #print(type(block))
    #print("############")
    res=[]
    for i in strand:
        res.append(copy_propagation(i))
    strand=res
    #pint(f"copy_propagation : {strand}")
    #print(type(strand))
    #print("############")
    res=[]
    for i in strand:
        res.append(dead_code_elimination(i))
    strand=res
    #print(f"dead_code_elimination : {strand}")
    #print(type(strand))
    #print(r"############")
    res=[]
    for i in strand:
        res.append(normalize(i))
    strand=res
    #print(f"normalize : {strand}")
    #print(type(strand))
    #print("############")
    vector=extract_vector_from_strand(strand,size)
    #print(vector)
    return vector




def create_pair_result(res_arr,result_folder,res_num):
    num=res_num
    for idx1 in range(len(res_arr)):
        for idx2 in range(len(res_arr)):
            #print(len(res_arr),idx1,idx2,get_str(idx1),get_str(idx2))
            new_folder=result_folder+'sample'+str(num)
            num+=1    
            #print(new_folder)
            #print(f"new folder: {new_folder}")
            os.mkdir(new_folder)
            #print("mkdir")
            #print(idx1,idx2,get_str(idx1),get_str(idx2))
            w1=open(new_folder+'/'+'file1'+'.txt','wb')
            w1.write(str(res_arr[idx1]).encode())
            w1.close()
            w2=open(new_folder+'/'+'file2'+'.txt','wb')
            w2.write(str(res_arr[idx2]).encode())
            w2.close()
            print(f"Created similarity pari function: {num}")


options=['O0', 'O1','O2','O3','Os']
result_path='/home/hai20521281/Downloads/TEST/result_similar/'
dataset = DatasetBase('/home/hai20521281/Downloads/TEST/sample', None, True, ['O0', 'O1','O2','O3','Os'])
dataset.load_pair_data()

cnt=0
ft_dataset = dataset.get_paired_data()
print("loaded")
ff={}
for proj, func_name, func_data in ft_dataset:
    res=[]
    try: 
        for opt in options:
            func_addr, asm_list, rawbytes_list, cfg, biai_featrue = func_data[opt]
            #print(func_name, rawbytes_list)
            #print('---------------------------')
            #print(rawbytes_list)
            res.append(process_a_func(rawbytes_list,1024))
            
        #print(res)
        #print(res)
        create_pair_result(res,result_path,cnt)
        #create_pair_result(res_256,result_path_256,cnt)
        #create_pair_result(res_512,result_path_512,cnt)
        #create_pair_result(res_1024,result_path_1024,cnt)
        cnt+=25
        if proj not in ff:
            ff[proj]=1
        else:
            ff[proj]+=1
    except Exception as ex:
        print(ex)
    #print(func_data)
    print("##############################")
print(cnt)

print(ff)








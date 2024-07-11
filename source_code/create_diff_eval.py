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



def parse_full_filename_to_file_name(filename):
    return "-".join(filename.split("/")[-1].split('-')[:-1])


def save_res_diff(vector1,vector2,result_folder,filename1,filename2,func_name1,func_name2):
    new_folder=result_folder+parse_full_filename_to_file_name(filename1)+'--'+func_name1+'___'+parse_full_filename_to_file_name(filename2)+'--'+func_name2
    os.mkdir(new_folder)
    w1=open(new_folder+'/'+parse_full_filename_to_file_name(filename1)+'--'+func_name1+'.txt','wb')
    w1.write(str(vector1).encode())
    w1.close()
    w2=open(new_folder+'/'+parse_full_filename_to_file_name(filename2)+'--'+func_name2+'.txt','wb')
    w2.write(str(vector2).encode())
    w2.close() 


dif_result_path='/home/hai20521281/Downloads/TEST/result_diff_eval/'

sample_dir='/home/hai20521281/Downloads/TEST/sample_eval/'
list_folder=os.listdir(sample_dir)

arr_file_name=[]
for folder in list_folder:
    list_filename=os.listdir(sample_dir+folder)
    for file_name in list_filename:
        if file_name=="saved_index.pkl":
            pass
        else:
            arr_file_name.append(sample_dir+folder+"/"+file_name)



def save_res_diff(vector1,vector2,result_folder,i):
    new_folder=result_folder+'sample_diff'+str(i)
    os.mkdir(new_folder)
    w1=open(new_folder+'/'+'file1'+'.txt','wb')
    w1.write(str(vector1).encode())
    w1.close()
    w2=open(new_folder+'/'+'file2'+'.txt','wb')
    w2.write(str(vector2).encode())
    w2.close() 

count_target=2474
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
        
                current_vector=process_a_func(data_pkl[function][-3],1024)
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
                        another_vector=process_a_func(another_file_pkl[another_function][-3],1024)
                        #print(another_file_pkl[another_function][-3])
                        save_res_diff(current_vector,another_vector,dif_result_path,count_result)
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

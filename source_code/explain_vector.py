import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pyvex
import archinfo
from analyse import *

# Function to convert raw bytes to VEX IR
def get_vexIR(raw_bytes):
    irsb = pyvex.IRSB(data=raw_bytes, mem_addr=0x08048429, arch=archinfo.ArchX86())
    result = b''
    for stmt in irsb.statements:
        if not isinstance(stmt, pyvex.IRStmt.IMark):
            result += (str(stmt).encode() + b'\n')
    return result.decode()

# Process function to extract features
def process_a_func(raw_bytes, size):
    print("Raw bytes input:")
    print(raw_bytes)

    vex_ir = get_vexIR(raw_bytes)
    print("VEX IR:")
    print(vex_ir)

    block = vex_ir.split("\n")[:-1]
    strand = parse_bb_to_strand(block)
    print("Parsed strand:")
    print(strand)

    res = []
    for i in strand:
        res.append(copy_propagation(i))
    strand = res
    print("After copy propagation:")
    print(strand)

    res = []
    for i in strand:
        res.append(dead_code_elimination(i))
    strand = res
    print("After dead code elimination:")
    print(strand)

    res = []
    for i in strand:
        res.append(normalize(i))
    strand = res
    print("After normalization:")
    print(strand)

    # Constant Folding
    res = []
    for i in range(len(block)):
        try:
            if isinstance(block[i], pyvex.IRStmt.IMark) and isinstance(block[i].expr, pyvex.expr.Const):
                res.append(block[i])
        except Exception as e:
            pass
    strand.extend(res)
    print("After Constant Folding:")
    print(strand)

    # Inlining
    res = []
    for i in range(len(block)):
        try:
            if isinstance(block[i], pyvex.IRStmt.IMark) and isinstance(block[i].expr, pyvex.expr.BinOp) and block[i].expr.op in [pyvex.IRExpr.Op.Add, pyvex.IRExpr.Op.Sub, pyvex.IRExpr.Op.Mul, pyvex.IRExpr.Op.UDiv]:
                # Replace BinOp expression with its constant result
                constant_result = block[i].expr.eval()
                block[i] = pyvex.IRStmt.IMark(pyvex.expr.Const(constant_result))
                res.append(block[i])
        except Exception as e:
            pass
    strand.extend(res)
    print("After Inlining:")
    print(strand)

    vector = extract_vector_from_strand(strand, size)
    print("Final vector:")
    print(vector)

    return vector

# Define the two functions as byte sequences
function1 = b'\xf3\x0f\x1e\xfaH\x8bG\x18\xc3'
function2 = b'\xf3\x0f\x1e\xfaUH\x89\xe5H\x89}\xf8H\x8bE\xf8H\x8b@\x18]\xc3'

# Process the functions and extract vectors
vector_func1 = process_a_func(function1, 1024)
vector_func2 = process_a_func(function2, 1024)

print("Processed function1 to vector: ")
print(vector_func1)
print("Processed function2 to vector: ")
print(vector_func2)


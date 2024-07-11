import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import pyvex
import archinfo
from analyse import *

model = tf.keras.models.load_model('model_binshoo.h5')

def are_functions_similar(function1, function2, threshold=0.5):
  target = []
  target.append(function1)
  target.append(function2)
  # Assuming your model expects an input shape of (None, 2, 1024)
  target = np.reshape(target, (1, 2, 1024))
  similarity_score = model.predict(target)[0][1]
  return similarity_score >= threshold

def get_vexIR(raw_bytes):
  irsb = pyvex.IRSB(data=raw_bytes, mem_addr=0x08048429, arch=archinfo.ArchX86())
  result = b''
  for stmt in irsb.statements:
    if not isinstance(stmt, pyvex.IRStmt.IMark):
      result += (str(stmt).encode() + b'\n')
  return result.decode()

def process_a_func(raw_bytes, size):
    print("Raw bytes input:")
    print(raw_bytes)

    vex_ir = get_vexIR(raw_bytes)

    block = vex_ir.split("\n")[:-1]
    strand = parse_bb_to_strand(block)

    res = []
    for i in strand:
        res.append(copy_propagation(i))
    strand = res

    res = []
    for i in strand:
        res.append(dead_code_elimination(i))
    strand = res
    res = []
    for i in strand:
        res.append(normalize(i))
    strand = res

    vector = extract_vector_from_strand(strand, size)
    return vector

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_function1', type=str, default=None)
  parser.add_argument('--path_function2', type=str, default=None)
  parser.add_argument('--threshold', type=float, default=0.5)  # Set your desired threshold

  args = parser.parse_args()
  function1 = open(args.path_function1, 'rb').read()
  function2 = open(args.path_function2, 'rb').read()

  # Assuming your model expects an input shape of (None, 2, 1024)
  vector_func1 = process_a_func(function1, 1024)
  vector_func2 = process_a_func(function2, 1024)

  print("Processed function1 to vector: ")
  print(vector_func1)
  print("Processed function2 to vector: ")
  print(vector_func2)

  similar = are_functions_similar(vector_func1, vector_func2, threshold=args.threshold)

  if similar:
      print("Functions are similar.")
  else:
      print("Functions are not similar.")


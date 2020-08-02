from pyspark import SparkContext, StorageLevel
from collections import Counter
from operator import add
import json
import os
import time
import sys
import itertools
import math
import csv
import string

# variables
input_file_path = sys.argv[1]
model = sys.argv[2]
output_file_path = sys.argv[3]


# ===================================================== START ===================================================
print("____________________ HEY ")

SparkContext.setSystemProperty('spark.executor.memory', '15g')
SparkContext.setSystemProperty('spark.driver.memory', '15g')
sc = SparkContext('local[*]', 'task2test')
start = time.time()
with open(model,'r') as f:
    mod_file = [json.loads(line) for line in f]
similarity = {}
for x in mod_file:
    for k,v in x.items():
      similarity[k] = v

print ("============================ similarity calculated")
    

def cossimilarity(puro):
    userid = puro[0]
    bizid = puro[1]
    if bizid not in similarity:
        return 0
    w = set(similarity[bizid])
    v = set(similarity[userid])
    numerator = len(w & v)
    denominator = math.sqrt(len(w)) * math.sqrt(len(v))
    cossim = numerator / denominator
    return cossim


test_op = sc.textFile(input_file_path).map(lambda x: (json.loads(x)['user_id'], json.loads(x)['business_id'])).map(lambda x: ((x[0], x[1]), cossimilarity(x))).filter(lambda x: x[1] >= 0.01).collect()

with open(output_file_path,'w') as f:
    for x in test_op:
        res_dic = {"user_id":x[0][0],"business_id":x[0][1],"sim":x[1]}
        json.dump(res_dic,f)
        f.write('\n')
        


end = time.time()
print("\n\nDuration:", end - start)
print("____________________ BYE")
# =================================================== END =================================================
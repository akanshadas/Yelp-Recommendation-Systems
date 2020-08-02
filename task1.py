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

# variables



input_file_path = sys.argv[1]
output_file_path = sys.argv[2]


# ========================= STEP 1: CREATING A SIGNATURE MATRIX =========================

def creatingSignatureMatrix(no_of_h, users , no_of_users):
    m = no_of_users + 1
    signatureRow = list()
    for i in range (no_of_h):
        signatureRow.append(m)

    y_limit = no_of_h + 1
    for user in list(users):
        for a in range(1, y_limit):
            b = a - 1; h = ((3233*a) + (a * user)) % no_of_users
            if signatureRow[b] > h: signatureRow[b] = h
    return signatureRow

# =================================== STEP 2: LSH ===================================
def bandComparison(signature1, signature2, B, R, n):
    i = 0
    while i < n:
        a = signature1[i:i + R]
        b = signature2[i:i + R]
        if a == b:
            return True
        i = i + R

    return False


def LSH(B, R, business_id, signatures):
    signatures = list(signatures)
    ins = 0
    signature_tuples = list()
    for ob in range(0, B):
        # dividing signs into rows of 2 for each band
        start_limit = ob * R; end_limit = (ob * R) + R;
        final_signature = signatures[start_limit:end_limit]
        final_signature.insert(ins, ob)
        tup = tuple(final_signature)
        signature_tuples.append((tup, business_id))
    a = signature_tuples
    return a

============================== STEP 3: JACCARD SIMILARITY ==============================
def jaccardSimilarityCalculation(cand1, cand2):
    intersection = len(a & b)
    union = len(a | b)

    sim = float(intersection) / float(union)
    return sim
	
def JS(candidate, business_data):
    a1 = business_data[candidate[1]]
    a = set(a1)
    b1 = business_data[candidate[0]]
    b = set(b1)
    jacc_sim = float(len(b.intersection(a))) / float(len(b.union(a)))
    return (candidate, jacc_sim)	

# ===================================================== START ===================================================

SparkContext.setSystemProperty('spark.executor.memory', '4g')
SparkContext.setSystemProperty('spark.driver.memory', '4g')
sc = SparkContext('local[*]', 'task1')
#sc.setLogLevel("ERROR")
start = time.time()

user_businessRDD = sc.textFile(input_file_path).map(lambda e: (json.loads(e)['user_id'], json.loads(e)['business_id'])).persist()

user_mapping_dict = user_businessRDD.map(lambda entry: entry[0]).zipWithIndex().collectAsMap()  
num_users = user_businessRDD.map(lambda x: x[0]).distinct().count()
business_user_map = user_businessRDD.map(lambda x: (x[1], user_mapping_dict[x[0]])).groupByKey().sortByKey()

# STEP 1: MINHASHING
no_of_hashes = 64

signatureMatrix = business_user_map.mapValues(lambda x: creatingSignatureMatrix(no_of_hashes, x, num_users))  

# STEP 2: LOCALITY SENSITIVE HASHING
B = 64
R = 1

resultOfLSH = signatureMatrix.flatMap(lambda x: LSH(B, R, x[0], x[1])).groupByKey()
resultOfLSH = resultOfLSH.filter(lambda x: len(list(x[1])) > 1)

# STEP 3: JACCARD SIMILARITY
business_data = {};
b = business_user_map.map(lambda x:(x[0], set(x[1]))).collect()
for o in b:
    business_data.update({o[0]: o[1]})



simPairs = resultOfLSH.flatMap(lambda x: sorted(list(itertools.combinations(sorted(list(x[1])), 2)))).distinct().map(lambda cd: JS(cd, business_data))
s = simPairs.filter(lambda x: x[1] >= 0.05)

result = s.sortByKey().collect()

with open(output_file_path, "w") as f:
    for x in result:
        output = {"b1": x[0][0], "b2": x[0][1], "sim": x[1]}
        json.dump(output, f)
        f.write('\n')

end = time.time()
print("\n\nDuration:", end - start)
print("____________________ BYE")
# =================================================== END =================================================
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
# input_file_path = sys.argv[1]
input_file_path = sys.argv[1]
# input_file_path = "train_review_task3.json"
output_file_path = sys.argv[2]
# output_file_path = "task3item.model"
cf_type = sys.argv[3]

SparkContext.setSystemProperty('spark.executor.memory', '15g')
SparkContext.setSystemProperty('spark.driver.memory', '15g')
sc = SparkContext('local[*]', 'task3train')
# sc.setLogLevel("ERROR")
start = time.time()

if cf_type == "user_based":

    # ================================ STEP 1: MINHASH - LSH - JACCARD =============================

    # ============================= STEP 1a: CREATING A SIGNATURE MATRIX ===========================
    def creatingSignatureMatrix(no_of_h, users, no_of_users):
        # users = ('S1', [6, 8]), ('S2', [7]), ('S3', [3, 8, 5]), ('S4', [6, 7, 8])
        # users = [6, 8], [7], [3, 8, 5], [6, 7, 8]
        # system_max_value = sys.maxsize  #9223372036854775807
        m = no_of_users + 1  # KINI, try this.
        signatureRow = list()
        for i in range(no_of_h):
            signatureRow.append(m)

        y_limit = no_of_h + 1
        for user in list(users):
            for a in range(1, y_limit):
                b = a - 1;
                h = ((3233 * a) + (a * user)) % no_of_users
                if signatureRow[b] > h: signatureRow[b] = h
        return signatureRow


    # ===================================== STEP 1b: LSH =====================================
    def LSH(B, R, business_id, signatures):
        signatures = list(signatures)
        ins = 0
        signature_tuples = list()
        for ob in range(0, B):
            # dividing signs into rows of 2 for each band
            start_limit = ob * R;
            end_limit = (ob * R) + R;
            final_signature = signatures[start_limit:end_limit]
            final_signature.insert(ins, ob)
            tup = tuple(final_signature)
            signature_tuples.append((tup, business_id))
        a = signature_tuples
        return a


    # ============================= STEP 1c: JACCARD SIMILARITY =============================
    def JaccardSim(candidate, business_data):
        a1 = business_data[candidate[1]]
        a = set(a1)
        b1 = business_data[candidate[0]]
        b = set(b1)

        num = len(b.intersection(a))
        denom = len(b.union(a))

        jacc_sim = float(num) / float(denom)
        # condition 2: if jaccard similarity >= 0.01:
        if (jacc_sim < 0.01):
            return (None, None)
        return (candidate, jacc_sim)


    # ====================== STEP 2: FIND TOPSIMILAR USERS + PEARSON SIMILARITY =============
    def user_pearsonCorrelation(activeUserData, otherUserData):
        activeUserData.sort()
        otherUserData.sort()

        a_corr_items = [];
        o_corr_items = []
        a = 0;
        o = 0
        len_a = len(activeUserData)
        len_o = len(otherUserData)
        while (a < len_a and o < len_o):
            if activeUserData[a][0] == otherUserData[o][0]:
                a_corr_items.append(activeUserData[a][1])
                o_corr_items.append(otherUserData[o][1])
                a += 1;
                o += 1
            elif activeUserData[a][0] < otherUserData[o][0]:
                a += 1
            else:
                o += 1

        if len(a_corr_items) < 3:  # we need at least 3 corrated items
            return "nan"  # KINI ADD THIS BACK

        r_a = sum(a_corr_items) / len(a_corr_items)
        r_o = sum(o_corr_items) / len(o_corr_items)

        num = 0
        denom_a = 0
        denom_o = 0
        for i in range(len(a_corr_items)):
            Rui = a_corr_items[i] - r_a;
            Ruo = o_corr_items[i] - r_o
            num += Rui * Ruo
            denom_a += Rui * Rui
            denom_o += Ruo * Ruo

        if num == 0 or denom_a == 0 or denom_o == 0:
            return "nan"  # check this
        result = num / (math.sqrt(denom_a) * math.sqrt(denom_o))

        if (result < 0):
            return "nan"

        return result


    # ======================================= START ====================================
    no_of_hashes = 64
    B = 64
    R = 1
    start = time.time()
    # STEP 1: FIND SIMILAR USERS by calculating Minhash - LSH - Jaccard Sim >= 0.01

    user_businessRDD = sc.textFile(input_file_path).map(lambda e: (json.loads(e)['business_id'], json.loads(e)['user_id'], json.loads(e)['stars'])).persist()
    user_mapping_dict = user_businessRDD.map(lambda entry: entry[0]).zipWithIndex().collectAsMap()
    num_users = user_businessRDD.map(lambda x: x[0]).distinct().count()
    data = user_businessRDD.map(lambda x: (x[1], user_mapping_dict[x[0]])).groupByKey().sortByKey()

    # STEP 1: MINHASHING
    signatureMatrix = data.mapValues(lambda x: creatingSignatureMatrix(no_of_hashes, x, num_users)) 

    # STEP 2: LOCALITY SENSITIVE HASHING
    resultOfLSH = signatureMatrix.flatMap(lambda x: LSH(B, R, x[0], x[1])).groupByKey()
    resultOfLSH = resultOfLSH.filter(lambda x: len(list(x[1])) > 1)

    # STEP 3: JACCARD SIMILARITY
    business_data = {};
    b = data.map(lambda x: (
    x[0], set(x[1]))).collect() 
    for bu in b:
        business_data.update({bu[0]: bu[1]})

    simPairs = resultOfLSH.flatMap(
        lambda x: sorted(list(itertools.combinations(sorted(list(x[1])), 2)))).distinct().map(
        lambda cd: JaccardSim(cd, business_data)).filter(lambda x: x[1] is not None)

    # STEP 2: FIND TOPSIMILAR USERS + PEARSON SIMILARITY
    users = user_businessRDD.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(list).collectAsMap()
    ps = simPairs.map(lambda x: (x[0][0], x[0][1])).map(
        lambda x: (x[0], x[1], user_pearsonCorrelation(users[x[0]], users[x[1]]))).filter(
        lambda x: x[2] != "nan").collect()
    with open(output_file_path, 'w', newline='') as f:
        for its in ps:
            f.write("{\"u1\": \"" + str(its[0]) + "\", \"u2\": \"" + str(its[1]) + "\", \"sim\": " + str(its[2]) + "}")
            f.write('\n')

# =================================== ITEM BASED FUNCTIONS ================================

if cf_type == "item_based":

    def pearsonCorrelation(activeItemData, otherItemData):
        a_corr_items = [];
        o_corr_items = []
        a = 0;
        o = 0
        len_a = len(activeItemData)
        len_o = len(otherItemData)
        while (a < len_a and o < len_o):
            if activeItemData[a][0] == otherItemData[o][0]:
                a_corr_items.append(activeItemData[a][1])
                o_corr_items.append(otherItemData[o][1])
                a += 1;
                o += 1
            elif activeItemData[a][0] < otherItemData[o][0]:
                a += 1
            else:
                o += 1
        if len(a_corr_items) < 1:
            return "nan"               

        r_a = sum(a_corr_items)/len(a_corr_items)
        r_o = sum(o_corr_items)/len(o_corr_items)

        num = 0
        denom_a = 0
        denom_o = 0
        for i in range (len(a_corr_items)):
            Rui = a_corr_items[i] - r_a; Ruo = o_corr_items[i] - r_o
            num += Rui*Ruo
            denom_a += Rui * Rui
            denom_o += Ruo * Ruo
        if num == 0 or denom_a == 0 or denom_o == 0:
            return "nan"                         
        result = num/(math.sqrt(denom_a) * math.sqrt(denom_o))
        return result
		
    def calculatingSimilarItems(activeItem, activeItemData):
        listOfSimilarItems = []
        if (activeItem not in items):               #cold start problem. Nobody has rated this item
            return [("nan", "nan")]

        #other items
        for otherItem in items:
             if otherItem > activeItem :
                otherItemData = items[otherItem]

                #Calculating pearson correlation
                corratedItems = []
                similarity = pearsonCorrelation(activeItemData, otherItemData)
                if similarity != "nan":
                    listOfSimilarItems.append((similarity, otherItem))
        if (len(listOfSimilarItems) == 0):
            return [("nan", "nan")]
        return listOfSimilarItems

    # ================================ START ================================
    trainingFile = sc.textFile(input_file_path).map(lambda e: (json.loads(e)['user_id'], json.loads(e)['business_id'], json.loads(e)['stars']))

    itemsRDD = trainingFile.map(lambda e: (e[1], (e[0], e[2]))).groupByKey().mapValues(list).persist() 
	
    #finding Pearson similarity
    ps = itemsRDD.map(lambda e:(e[0], calculatingSimilarItems(e[0], e[1]))).filter(lambda e: e[1] != [('nan', 'nan')]).map(lambda e: (e[0], [x for x in e[1] if x[0] > 0])).collect()

    with open(output_file_path, 'w', newline='') as f:
        for its in ps:
            for it in its[1]:
                f.write("{\"b1\": \"" + str(its[0]) + "\", \"b2\": \"" + str(it[1]) + "\", \"sim\": " + str(it[0]) + "}")
                f.write('\n')
end = time.time()
print("\n\nDuration:", end - start)
print("____________________ BYE")
# ================================ END ==================================

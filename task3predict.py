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

train_file = sys.argv[1]
test_file = sys.argv[2]
model_file = sys.argv[3]
output_file_path = sys.argv[4]
cf_type = sys.argv[5]

def prediction(activeItem, otherItemRatings):       
    N = 1000
    
    if len(otherItemRatings) == 0:        #cold start problem - activeUser has not rated any items, is a new user
        return "nan"
    user_ratings = [x[1] for x in otherItemRatings]
    user_mean = sum(user_ratings)/len(user_ratings)
        
    #find the top N Ws
    topSimItems = []
    for o_item, o_rating in otherItemRatings:
        if (activeItem, o_item) in modelTrain:
            topSimItems.append((o_rating, modelTrain[(activeItem, o_item)]))
        elif (o_item, activeItem) in modelTrain:
            topSimItems.append((o_rating, modelTrain[(o_item, activeItem)]))

    topNSimItems = sorted(topSimItems, key=lambda x: (-x[1]))[:N]
    
    if len(topNSimItems) > 0 and topNSimItems[0][1] < 0.01: 
        return "nan"
    if len(topNSimItems) < 2:                #cold start problem - item not rated by other users 
        return "nan"
    else:
        num = 0
        denom = 0

        for rating, sim in topNSimItems:
            num += rating * sim
            denom += abs(sim)

        if num == 0 or denom == 0:
            return "nan"

    pred = num/denom
    if pred < 0:
        pred = pred * -1
    #print ("PRED:", pred)
    return pred
           
# ===================================================== START ===================================================
print("____________________ HEY ")

SparkContext.setSystemProperty('spark.executor.memory', '15g')
SparkContext.setSystemProperty('spark.driver.memory', '15g')
sc = SparkContext('local[*]', 'task3predict')
start = time.time()

trainingRDD = sc.textFile(train_file).map(lambda e: (json.loads(e)['user_id'], (json.loads(e)['business_id'], json.loads(e)['stars'])))
testRDD = sc.textFile(test_file).map(lambda e: (json.loads(e)['user_id'], json.loads(e)['business_id']))

modelTrain = sc.textFile(model_file).map(lambda e: ((json.loads(e)['b1'], json.loads(e)['b2']), json.loads(e)['sim'])).collectAsMap()
j = testRDD.join(trainingRDD).map(lambda e: ((e[0], e[1][0]), e[1][1])).groupByKey().mapValues(list).map(lambda e: (e[0], prediction(e[0][1], e[1])) ).filter(lambda e: e[1] != "nan")
pred = j.collect()


end = time.time();
print("\n\nDuration:", end - start)

with open(output_file_path, 'w', newline='') as f:
    for p in pred:
        f.write("{\"user_id\": \"" + str(p[0][0]) + "\", \"business_id\": \"" + str(p[0][1]) + "\", \"stars\": " + str(p[1]) + "}")
        f.write('\n')

end = time.time()
print("\n\nDuration:", end - start)
print("____________________ BYE")
# =================================================== END =================================================

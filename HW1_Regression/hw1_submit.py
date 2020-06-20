import numpy as np
import pandas as pd
import sys
import math
import csv
import matplotlib.pyplot as plt
import pickle
#load
test_file = sys.argv[1]
output_file = sys.argv[2]
testdata = pd.read_csv(test_file, encoding = 'big5')
# Preprocess
testdata = testdata.iloc[:, 2:]
testdata[testdata == 'NR'] = 0
test_data = testdata.to_numpy()
test_x = np.empty([240, 162], dtype = float)
remain = []
delete = np.zeros((162, 1))
for i in range(9):
    delete[i] = 1
    delete[126+i] = 1
    delete[135+i] = 1
    delete[144+i] = 1
    delete[153+i] = 1
for i in range(18*9):
    if delete[i] == 0:
        remain.append(i)
mean_x = np.load('mean_x.npy')
std_x = np.load('std_x.npy')
test_data = np.concatenate((np.ones([1, 9]), test_data), axis = 0).astype(float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
test_x = test_x[:,remain]
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)   
w = np.load('weight.npy')
# predict
ans_y = np.dot(test_x, w)
for i in range(len(ans_y)):
    if ans_y[i] < 0:
        ans_y[i] = 0
with open(output_file, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
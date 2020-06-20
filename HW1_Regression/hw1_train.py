import numpy as np
import pandas as pd
import sys
import math
import csv
import matplotlib.pyplot as plt
import pickle
iter_time = 10000   
learning_rate = 1
#load
train_file = sys.argv[1]
test_file = sys.argv[2]
data = pd.read_csv(train_file, encoding = 'big5')
testdata = pd.read_csv(test_file, encoding = 'big5')
# Preprocess
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
testdata = testdata.iloc[:, 2:]
testdata[testdata == 'NR'] = 0
test_data = testdata.to_numpy()
test_x = np.empty([240, 162], dtype = float)

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample
x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
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
x = x[:,remain]
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9
np.save('mean_x.npy', mean_x)
np.save('std_x.npy', std_x)
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
test_x = test_x[:,remain]
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)   
x_train_set = x
y_train_set = y
#[:math.floor(len(x) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
rx, cx = x_train_set.shape
vrx, vcx = x_validation.shape
dim = cx+1
x_train_set = np.concatenate((np.ones([rx, 1]), x_train_set), axis = 1).astype(float)
x_validation = np.concatenate((np.ones([vrx, 1]), x_validation), axis = 1).astype(float)
w = np.zeros([dim, 1])
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
train_data = []
val_data = []
for t in range(iter_time):
    train_loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/rx)
    val_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/vrx)
    if(t%1000==0):
        #print('{}, train_loss={}, val_loss={}'.format(t, train_loss, val_loss))
        train_data.append(train_loss)
        val_data.append(val_loss)
    #update w
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
# predict
ans_y = np.dot(test_x, w)
np.save('weight.npy', w)
with open('W_0324.csv', mode='w', newline='') as f:
    for i in range(13):
        for j in range(9):
            f.write('{},'.format(str(w[i*9+j+1])))
        f.write('\n')
    f.write('{},'.format(str(w[13*9])))
for i in range(len(ans_y)):
    if ans_y[i] < 0:
        ans_y[i] = 0
with open('submit_0324.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
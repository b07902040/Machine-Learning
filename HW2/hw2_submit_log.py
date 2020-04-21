import numpy as np
import math
import sys
#np.set_printoptions(threshold=sys.maxsize)
dev_ratio = 0.1
max_iter = 100
batch_size = 8
learning_rate = 0.01
big_loop = 20
train_num = 54255
seed1, seed2 = 88, 863  
lamda = 0.01
#print(max_iter, batch_size, learning_rate, seed1, seed2)
#print('Update times: {}'.format(big_loop*max_iter*int((train_num*(1-dev_ratio)/batch_size))))
X_train_fpath = sys.argv[3]
Y_train_fpath = sys.argv[4]
X_test_fpath = sys.argv[5]
output_fpath = sys.argv[6]
with open(X_train_fpath) as f:
    next(f)
    Original_X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Original_Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    Original_X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
delete_list = [5, 127, 148, 169, 193, 199, 221, 255, 323, 330, 334, 343, 351, 354, 359, 501, 261, 331, 340, 350, 355, 394, 438, 481]
Original_X_train = np.delete(Original_X_train, delete_list, axis=1)
Original_X_test = np.delete(Original_X_test, delete_list, axis=1)
# Normalize training and testing data
def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std
Original_X_train, Original_X_mean, Original_X_std = _normalize(Original_X_train, train = True)
Original_X_test, _, _= _normalize(Original_X_test, train = False, specified_column = None, X_mean = Original_X_mean, X_std = Original_X_std)
# Functions
def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))
def _f(X, w, b):
    return _sigmoid(np.matmul(X, w) + b)
def _predict(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)
def Predict(w, b):
    predictions = _predict(Original_X_test, w, b)
    with open(output_fpath, 'w') as f:
        f.write('id,label\n')
        for i, label in enumerate(predictions):
            f.write('{},{}\n'.format(i, label))
if __name__ == '__main__':
    maxw = np.load('weight_log.npy')
    maxb = np.load('b_log.npy')
    Predict(maxw, maxb)

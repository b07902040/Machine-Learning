import numpy as np
import sys
def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std
def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))
def _f(X, w, b):
    return _sigmoid(np.matmul(X, w) + b)
def _predict(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

np.random.seed(0)
X_train_fpath = sys.argv[3]
Y_train_fpath = sys.argv[4]
X_test_fpath = sys.argv[5]
output_fpath = sys.argv[6]

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
delete_list = [5, 127, 148, 169, 193, 199, 221, 255, 323, 330, 334, 343, 351, 354, 359, 501, 261, 331, 340, 350, 355, 394, 438, 481]
X_train = np.delete(X_train, delete_list, axis=1)
X_test = np.delete(X_test, delete_list, axis=1)
# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

w = np.load('weight_gen.npy')
b = np.load('b_gen.npy')

# Compute accuracy on training set
Y_train_pred = 1 - _predict(X_train, w, b)
#-------
# Predict testing labels
predictions = 1 - _predict(X_test, w, b)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))


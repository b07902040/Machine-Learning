import numpy as np
import matplotlib.pyplot as plt
import math
#np.set_printoptions(threshold=sys.maxsize)
dev_ratio = 0.1
max_iter = 100
batch_size = 8
learning_rate = 0.01
big_loop = 20
train_num = 54255
seed1, seed2 = 88, 863  
lamda = 0.01
print(max_iter, batch_size, learning_rate, seed1, seed2)
print('Update times: {}'.format(big_loop*max_iter*int((train_num*(1-dev_ratio)/batch_size))))

X_train_fpath = './X_train'
Y_train_fpath = './Y_train'
X_test_fpath = './X_test'
output_fpath = './output_{}.csv'
'''
X_train_fpath = '/kaggle/input/X_train'
Y_train_fpath = '/kaggle/input/Y_train'
X_test_fpath = '/kaggle/input/X_test'
output_fpath = './output_{}.csv'
'''
with open(X_train_fpath) as f:
    next(f)
    Original_X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Original_Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    Original_X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
#not in universe:5, 127, 169, 193, 199, 221, 255, 323, 334, 343, 354, 359, 501
#not in universe or children:148
#not in universe under 1 year old:351
#not identifiable:330
#do not know:186
#?:261, 331, 340, 350, 355, 394, 438, 481
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
def _train_dev_split(X, Y, dev_ratio):
    # This function spilts data into training set and development set.
    train_size = len(X) * dev_ratio
    return X[:int(train_size*8)], X[int(train_size*8):int(train_size*9)], X[int(train_size*9):], Y[:int(train_size*8)], Y[int(train_size*8):int(train_size*9)], Y[int(train_size*9):]
def _shuffle(X, Y, seed):
    # This function shuffles two equal-length list/array, X and Y, together.
    np.random.seed(seed)
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))
def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)
def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc
def _cross_entropy_loss(y_pred, Y_label, w):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))+lamda*np.sum(w**2)
    return cross_entropy
def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)+2*lamda*w/len(w)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad
def Training(X_train, X_test, Y_train, iterep):    
    # Split data into training set and development set
    X_train, Y_train = _shuffle(X_train, Y_train, (iterep+1)*seed1)
    X_train, X_dev1, X_dev2, Y_train, Y_dev1, Y_dev2 = _train_dev_split(X_train, Y_train, dev_ratio)
    train_size = X_train.shape[0]
    dev1_size = X_dev1.shape[0]
    dev2_size = X_dev2.shape[0]
    test_size = X_test.shape[0]
    data_dim = X_train.shape[1]
    print('Size of training set: {}'.format(train_size))
    print('Size of development set: {},{}'.format(dev1_size, dev2_size))
    print('Size of testing set: {}'.format(test_size))
    print('Dimension of data: {}'.format(data_dim))
    # initialization for weights ans bias
    w = np.zeros((data_dim,)) 
    b = np.zeros((1,))
    train_loss = []
    dev1_loss = []
    dev2_loss = []
    train_acc = []
    dev1_acc = []
    dev2_acc = []
    step = 1
    # Iterative training
    for epoch in range(max_iter):
        # Random shuffle at the begging of each epoch
        X_train, Y_train = _shuffle(X_train, Y_train, (epoch+1)*seed2)
        # Mini-batch training
        for idx in range(int(np.floor(train_size / batch_size))):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]
            w_grad, b_grad = _gradient(X, Y, w, b)
            # gradient descent update
            # learning rate decay with time
            w = w - learning_rate/np.sqrt(step) * w_grad
            b = b - learning_rate/np.sqrt(step) * b_grad
            step = step + 1   
        # Compute loss and accuracy of training set and development set
        y_train_pred = _f(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(_accuracy(Y_train_pred, Y_train))
        train_loss.append(_cross_entropy_loss(y_train_pred, Y_train, w) / train_size)

        y_dev1_pred = _f(X_dev1, w, b)
        Y_dev1_pred = np.round(y_dev1_pred)
        dev1_acc.append(_accuracy(Y_dev1_pred, Y_dev1))
        y_dev2_pred = _f(X_dev2, w, b)
        Y_dev2_pred = np.round(y_dev2_pred)
        dev2_acc.append(_accuracy(Y_dev2_pred, Y_dev2))
        dev1_loss.append(_cross_entropy_loss(y_dev1_pred, Y_dev1, w) / dev1_size)
        dev2_loss.append(_cross_entropy_loss(y_dev2_pred, Y_dev2, w) / dev2_size)
        if epoch%5 is 0:
            print('iter=', epoch, 'train=', train_acc[-1], 'dev1=', dev1_acc[-1], 'dev2=', dev2_acc[-1])
    '''
    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))
    '''
    # Loss curve
    plt.plot(train_loss)
    plt.plot(dev1_loss)
    plt.plot(dev2_loss) 
    plt.title('Loss, time:{}'.format(iterep))
    plt.legend(['train', 'dev1', 'dev2'])
    plt.savefig('loss_shu_{}.png'.format(iterep))
    plt.cla()
    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(dev1_acc)
    plt.plot(dev2_acc)
    plt.title('Accuracy, time:{}'.format(iterep))
    plt.legend(['train', 'dev1', 'dev2'])
    plt.savefig('acc_shu_{}.png'.format(iterep))
    plt.cla()
    return w, b, (dev1_acc[-1]+dev2_acc[-1])/2, (dev1_loss[-1]+dev2_loss[-1])/2
def Predict(w, b):
    predictions = _predict(Original_X_test, w, b)
    with open(output_fpath.format('logistic_shuffle'), 'w') as f:
        f.write('id,label\n')
        for i, label in enumerate(predictions):
            f.write('{},{}\n'.format(i, label))

if __name__ == '__main__':
    max_acc = -1
    maxi = -1
    max_loss = -1
    maxw = np.zeros((487,)) 
    maxb = np.zeros((1,))
    with open(X_test_fpath) as f:
        content = f.readline().strip('\n').split(',')[1:]
    features = np.array(content)
    print(features[186])
    for i in range(big_loop):
        w, b, acc, loss = Training(Original_X_train, Original_X_test, Original_Y_train, i)
        print('i=', i, 'acc=', acc)
        print('-------------------------')
        if acc > max_acc:
            max_acc = acc
            maxw = w
            maxb = b
            maxi = i
            max_loss = loss
            print('Pick:',maxi, 'max_acc=', max_acc, 'max_loss=', max_loss)
            Predict(maxw, maxb)
            # Output W.csv
            #ind = np.argsort(np.abs(maxw))[::-1]
            with open('W.csv', 'w') as f:
                f.write('label,weight\n')
                for i in range(len(maxw)):
                    f.write('{},{}\n'.format(features[i], maxw[i]))
    print('Pick:',maxi, 'max_acc=',max_acc)
    #ind = np.argsort(np.abs(maxw))[::-1]

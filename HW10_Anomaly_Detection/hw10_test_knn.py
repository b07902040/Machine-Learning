import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from scipy.cluster.vq import vq, kmeans
from sklearn.decomposition import PCA
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import os
import sys
from sklearn import preprocessing

#os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
#train = np.load('train.npy', allow_pickle=True)
test = np.load(sys.argv[1], allow_pickle=True)
task = 'ae'
model_type = 'fcn'
feature = 'fcn_sha_256_n4'

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print('use cuda')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    #random.seed(seed)
same_seeds(0)

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.Linear(16, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(True),
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True), 
            nn.Linear(256, 32 * 32 * 3), 
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#x = train.reshape(len(train), -1)
y = test.reshape(len(test), -1)
n = 4
y_x = MiniBatchKMeans(n_clusters=n, random_state=689).fit(y)
y_cluster = y_x.predict(y)
zero_list = [i for i in y_cluster if i == 0]
one_list = [i for i in y_cluster if i == 1]
two_list = [i for i in y_cluster if i == 2]
three_list = [i for i in y_cluster if i == 3]
print(len(zero_list))
print(len(one_list))
print(len(two_list))
print(len(three_list))  
y_dist = []
for i in range(len(y_cluster)):
    if y_cluster[i] == 3:
        y_dist.append(1)
    else :
        y_dist.append(0)
y_pred = np.asarray(y_dist) 
y_mean = np.mean(y_pred)
# ------------------
batch_size = 64
data = torch.tensor(y, dtype=torch.float)
test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

model = torch.load(sys.argv[2], map_location='cuda')
model.eval()
reconstructed = list()
for i, data in enumerate(test_dataloader): 
    if model_type == 'cnn':
        img = data[0].transpose(3, 1).cuda()
    else:
        img = data[0].cuda()
    output = model(img)
    if model_type == 'cnn':
        output = output.transpose(3, 1)
    elif model_type == 'vae':
        output = output[0]
    reconstructed.append(output.cpu().detach().numpy())

reconstructed = np.concatenate(reconstructed, axis=0)
anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
y_pred_2 = anomality　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
y_mean_2 = np.mean(y_pred_2)

scaler = y_mean/y_mean_2
print(scaler)             
y_pred = (25*y_pred_2*scaler + y_pred)/26

#y_pred = np.asarray(y_pred)
with open(sys.argv[3], 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))
print('End')
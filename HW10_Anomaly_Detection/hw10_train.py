import numpy as np
from sklearn.cluster import MiniBatchKMeans
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

#os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
train = np.load(sys.argv[1], allow_pickle=True)
#test = np.load('test.npy', allow_pickle=True)
task = 'ae'
feature = 'fcn_sha'

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

num_epochs = 50
batch_size = 16
learning_rate = 1e-3
model_type = 'fcn'  

x = train
x = x.reshape(len(x), -1)
    
data = torch.tensor(x, dtype=torch.float)
train_dataset = TensorDataset(data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

model = fcn_autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_loss = np.inf
model.train()
for epoch in range(num_epochs):
    for data in train_dataloader:
        img = data[0].cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================save====================
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model, sys.argv[2])
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
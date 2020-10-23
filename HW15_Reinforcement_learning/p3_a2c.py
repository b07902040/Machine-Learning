import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(8, 128)
        
        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).double()
        state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
# Defaults parameters:
#    gamma = 0.99
#    lr = 0.02
#    betas = (0.9, 0.999)
#    random_seed = 543

render = False
gamma = 0.99
lr = 0.02
betas = (0.9, 0.999)
MAX_STEP = 10000

env = gym.make('LunarLander-v2')
policy = ActorCritic()
optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
print(lr,betas)

running_reward = 0
plot_list = []
for i_episode in range(1, 6000):
    state = env.reset()
    for t in range(MAX_STEP):
        action = policy(state)
        state, reward, done, _ = env.step(action)
        policy.rewards.append(reward)
        running_reward += reward
        if render and i_episode > 1000:
            env.render()
        if done:
            break
    optimizer.zero_grad()
    loss = policy.calculateLoss(gamma)
    loss.backward()
    optimizer.step()        
    policy.clearMemory()
    if i_episode % 30 == 0:
        running_reward /= 30
        plot_list.append(running_reward)
        print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
        running_reward = 0
plt.plot(plot_list)
plt.title("Total Rewards")
plt.show()
np.save('a2c', plot_list)
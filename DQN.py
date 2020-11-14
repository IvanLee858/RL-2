import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
directory = 'model/'

class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, action_dim)
      
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.eval_net = Net(self.state_space_dim, self.action_space_dim,)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.buffer = []
        self.steps = 0
        
    def act(self, s0):
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            a0 = self.actor(state).cpu().data.numpy().flatten()
        return a0

    def put(self, *transition):
        if len( self.buffer)==self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
        
    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return
        
        samples = random.sample( self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor( s0, dtype=torch.float)
        a0 = torch.tensor( a0, dtype=torch.long).view(self.batch_size, -1)
        r1 = torch.tensor( r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor( s1, dtype=torch.float)
        
        y_true = r1 + self.gamma * torch.max( self.eval_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        y_pred = self.eval_net(s0).gather(1, a0)
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

import gym
from IPython import display
import matplotlib.pyplot as plt
from env import ArmEnv
#from dqn import Agent

def plot(score, mean):
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(20,10))
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(score)
    plt.plot(mean)
    plt.text(len(score)-1, score[-1], str(score[-1]))
    plt.text(len(mean)-1, mean[-1], str(mean[-1]))

if __name__ == '__main__':

    env = ArmEnv()

    params = {
        'gamma': 0.9,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200, 
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 32,
        'state_space_dim': env.state_dim,
        'action_space_dim': env.action_dim,
        }
    agent = Agent(**params)

    score = []
    mean = []

    for episode in range(1000):
        s0 = env.reset()
        total_reward = 1
        while True:
            env.render()
            a0 = agent.act(s0)
            s1, r1, done= env.step(a0)
            
            if done:
                r1 = -1
                
            agent.put(s0, a0, r1, s1)
            
            if done:
                break

            total_reward += r1
            s0 = s1
            agent.learn()
            
        score.append(total_reward)
        mean.append( sum(score[-100:])/100)
        
        plot(score, mean)
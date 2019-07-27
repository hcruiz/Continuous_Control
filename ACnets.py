'''
This file contains the Actor and Critic networks used to solve the Reacher Unity ML-agent.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):

    def __init__(self, inputsize, action_size):
        super(Policy, self).__init__()
        hl1_size = 256
        hl2_size = 256
        self.fc1 = nn.Linear(inputsize, hl1_size)
        self.fc2 = nn.Linear(hl1_size, hl2_size)
        self.fc_out = nn.Linear(hl2_size, action_size)
        self.tanh = nn.Tanh()
        
        print('Actor has input size', inputsize,' hidden layer 1 size: ', hl1_size, ' hidden layer 2 size: ', hl2_size,
        ' action_size',action_size)
        
    def forward(self, x):
        
        #From 33-dim state -> 4 actions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.tanh(self.fc_out(x)) # action defined in [-1,1]

class Critic(nn.Module):

    def __init__(self, inputsize):
        super(Critic, self).__init__()
        hl1_size = 256
        hl2_size = 256
        self.fc1 = nn.Linear(inputsize, hl1_size)
        self.fc2 = nn.Linear(hl1_size, hl2_size)
        self.fc_out = nn.Linear(hl2_size, 1)
        
        print('Critic has input size', inputsize,' hidden layer 1 size: ', hl1_size, ' hidden layer 2 size: ', hl2_size)
        
    def forward(self, x):
        
        #From 33-dim state -> 1 state-value
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
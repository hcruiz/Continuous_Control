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
        self.std = nn.Parameter(torch.ones(1,action_size))
        print('Actor has input size', inputsize,' hidden layer 1 size: ', hl1_size, ' hidden layer 2 size: ', hl2_size,
        ' action_size',action_size)
        
    def forward(self, x, *args):
        
        #From 33-dim state -> 4 actions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean_action = self.tanh(self.fc_out(x)) # action defined in [-1,1]
        probability = torch.distributions.Normal(mean_action, self.std)
        if len(args)==0:
            action = probability.sample()
        elif len(args)==1:
            action = args[0]
        else:
            assert False, 'Too many arguments, please give only states (and actions if needed).'
        log_p = torch.sum(probability.log_prob(action),dim=-1)
        return log_p, action

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
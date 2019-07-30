# Continuous Control in RL using a PPO Actor-Critic approach
This repository trains the [Reacher Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) agent for Udacity's Deep Reinforcement Learning Nano-degree. The task is for a robotic arm with 2 
joints to reach and follow a ball moving around the arm. I use an Actor-Critic method with a PPO loss function [following this paper](https://arxiv.org/pdf/1707.06347.pdf).

## The Reacher Environment
The [environment](https://www.youtube.com/watch?v=2N9EoF6pQyE&feature=youtu.be) consists of a double-jointed arm that should move to target locations. A reward of +0.1 is given to the arm for every time step that the agent's tip is in the target position (marked by a sphere). Hence, the goal of the agent is to keep its tip at the target location as long as possible.
The observation space is a 33-dimensional space describing the position, rotation, velocity, and angular velocities of the arm. the action-space is 4-dimensional, corresponding to the torques applied to each of the two joints. The element in the action vector is limited to the interval between -1 and 1.
Moreover, the environment consists of 20 independent, non-interacting but identical agents, i.e. the environment samples 20 copies of the same agent in parallel. This is useful for algorithms like PPO, A3C, and D4PG that require multiple samples for each learning step.
The environment is considered solved when the average score (total reward) over the 20 agents and over 100 consecutive episodes is larger or equal +30.

__NOTE__: The environment ends it's episode after 1000 time steps, i.e. the maximum toral reward achievable is less than 100. 

## Getting Started

First, install Anaconda (python 3) and clone/download this repository (from terminal use the `git clone` command). To install all the required packages needed to run this task you can create an environment using the .yml file in this repository. Just run on your terminal

`conda env create -f environment.yml`

where *environment.yml* is either `drlnd_Win64.yml` or `drlnd_ubuntu18.yml`. This environment is based on the environment provided by Udacity for this project, with the addition of the specific [PyTorch](https://pytorch.org/) version that I required and the Unity environment. To activate the environment run `conda activate drlnd` and verify that the environment is installed correctly using `conda list`.

__NOTE__: I was able to run on both my machines; however, there might be compatibility issues with yours, so make sure you have the proper environment set up.

Finally, you have to download the Reacher Unity environment. There are different versions depending on your operating system, so please make sure you have the correct version of the environment. The files of the environment must be placed in the repository directory or, if 
placed somewhere else, the initialization of the environment in the notebook must contain the path to the environment.

__NOTE:__ The torch version in this environment assumes Windows 10 and __no CUDA__ installation. If you want to run the neural networks using CUDA, please make sure you install the proper PyTorch version found [here](https://pytorch.org/get-started/locally/). 

## Instructions

The code is structured as follows. There are two modules [ACnets.py](https://github.com/hcruiz/Continuous_Control/blob/master/ACnets.py) and [Training.py](https://github.com/hcruiz/Continuous_Control/blob/master/Training.py) containing all the necessary functions and classes. 
As the name suggests, ACnets contains the Actor and the Critic network classes. The Actor class, called Policy, is a stochastic policy that uses a neural network to estimate the mean of a Gaussian and then samples from this to return the action. Additionally, it returns the log-probability value of that action. 
The Training module contains the rest of the functions needed for training, e.g. the loss functions (for actor and critic), the training function and other helper functions like get_samples.
The user should open the jupyter notebook [Continuous_Control.ipynb](https://github.com/hcruiz/Continuous_Control/blob/master/Continuous_Control.ipynb) and simply run all cells to start training. 
For details on the implementation and the results, please have a look at the [Report](https://github.com/hcruiz/Continuous_Control/blob/master/Report.md).

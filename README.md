# Continuous Control in RL using a PPO Actor-Critic appraoch
This repository trains the Reacher Unity agent for Udacity's Deep Reinforcement Learning Nano-degree. The task is for a robotic arm with 2 
joints to reach and follow a ball moving around the arm. I use an Actor-Critic method with a PPO loss function.

## The Reacher Environment

## Getting Started

First, install Anaconda (python 3) and clone/download this repository (from terminal use the `git clone` command). To install all the 
required packages needed to run this task you can create an envarionment using the .yml file in this repository. Just run on your terminal

`conda env create -f environment.yml`

This environment is based on the environment provided by Udacity for this project, with the addition of the specific [PyTorch](https://pytorch.org/) 
version that I required and the Unity environment. To activate the environment run `conda activate drlnd` and verify that the environment 
is installed correctly using `conda list`.

Finally, you have to download the Reacher Unity environment. There are different versions depending on your operating system, so please 
make sure you have the correct version of the environment. The files of the environment must be placed in the repository directory or, if 
placed somewhere else, the initialization of the environment in the notebook must contain the path to the environment.

__NOTE:__ The torch version in this environment assumes Windows 10 and __no CUDA__ installation. If you want to run the neural networks 
using CUDA, please make sure you install the proper PyTorch version found [here](https://pytorch.org/get-started/locally/). 

## Instructions

The code is structured as follows.

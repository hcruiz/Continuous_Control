# Report for Unity's Reacher training using Actor-Critic RL Control

In this report, I describe the algorithm used to train Unity's Reacher environment with 20 agents. I chose the parallel multi-agent environment in order to use the Proximal Policy Optimization (PPO) algorithm with Generalized Advantage Estimation and two neuran networks serving as the actor and the critic.

## Learning Algorithm

This environment is solved using an Actor-Critic method with Generalized Advantage Estimates and a Proximal Policy Optimization target with a clipping theshold ![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon). More specific,our RL-agent is compossed of two neural networks, the actor/policy network and the critic network. Both are trained with the same data gathered in one episode from the 20 parallel environments but optimizing two different functions ![equation](https://latex.codecogs.com/gif.latex?L%5E%7B%5Cepsilon%7D_%7BPPO%7D) and a simple quadratic cost funtion (MSE) respectively,

![equation](https://latex.codecogs.com/gif.latex?L%5E%7B%5Cepsilon%7D_%7BPPO%7D%20%3D%20%5Cfrac%7B1%7D%7BM%7D%5Csum_%7Bt%2Ci%7Dmin%5Cleft%20%5B%20A%5Ei_t%5Cfrac%7B%5Cpi_%7Bnew%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%7B%5Cpi_%7Bold%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%2Cclip_%7B%5Cepsilon%7D%5Cleft%28%20A%5Ei_t%5Cfrac%7B%5Cpi_%7Bnew%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%7B%5Cpi_%7Bold%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%20%5Cright%20%29%20%5Cright%20%5D)

![equation](https://latex.codecogs.com/gif.latex?L_%7BMSE%7D%20%3D%20%5Cfrac%7B1%7D%7BMT%7D%5Csum_%7Bt%2Ci%7D%5Cleft%20%5C%7C%20V%28s_t%5Ei%29%20-%20%5Chat%7BR%7D_t%5Ei%20%5Cright%20%5C%7C%5E%7B2%7D)

where  ![equation](https://latex.codecogs.com/gif.latex?M%2C%20%5Cpi%28a%5Ei_t%7Cs%5Ei_t%29%2C%20V%28s_t%5Ei%29%20%2C%20%5Chat%7BR%7D_t%5Ei)  are the number of samples (=timepoints x agents), the policy, value function and the discounted rewards respectively.

The Policy network is a stochastic policy, meaning that, although the mean of the action is a deterministic function of the state, the action is selected randomly from a Gaussian distribution around this mean. It is important to note that it does not suffice to sample the action from a Gaussian to enhance exploration at the beginning and then decrease the noise gradually. This will result in an unstable learning procedure. Hence, the standard deviation of the Gaussian in the policy must also be learned via gradient descent. 
In addition, I chose a (truncated) Generalized Advantage Estimation with the discounting factor ![equation](https://latex.codecogs.com/gif.latex?%5Clambda),

![equation](https://latex.codecogs.com/gif.latex?A_t%20%3D%20%5Csum_%7Bi%3D0%7D%5E%7BT-%28t-1%29%7D%28%5Cgamma%5Clambda%29%5Ei%5Cdelta_%7Bt&plus;i%7D)

where ![equation](https://latex.codecogs.com/gif.latex?%5Cdelta_t%20%3D%20r_t%20&plus;%20%5Cgamma%20V%28s_%7Bt&plus;1%7D%29-%20V%28s_%7Bt%7D%29)

For more details the reader is referred to the [PPO paper](https://arxiv.org/pdf/1707.06347.pdf).

### Neural Network Architectures
Both the actor and the critic neural networks have the two layers with 256 nodes each. Both have ReLUs as activation functions and hence, the only difference is in their output. While the critic outputs a single linear node, the actor's output is 4-dimensional and is passed through a hyperbolic tangent before it is fed as the mean to a Normal distribution, from which the action is sampled. 

### Parameters

## Results

## Future Work

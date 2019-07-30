# Report for Unity's Reacher training using Actor-Critic RL Control

In this report, I describe the algorithm used to train Unity's Reacher environment with 20 agents. I chose the parallel multi-agent environment to use the Proximal Policy Optimization (PPO) algorithm with Generalized Advantage Estimation and two neural networks serving as the actor and the critic.

## Learning Algorithm

This environment is solved using an Actor-Critic method with Generalized Advantage Estimates and a Proximal Policy Optimization target with a clipping threshold  ![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon). More specific, our RL-agent is composed of two neural networks, the actor/policy network and the critic network. Both are trained with the same data gathered in one episode from the 20 parallel environments but optimizing two different functions ![equation](https://latex.codecogs.com/gif.latex?L%5E%7B%5Cepsilon%7D_%7BPPO%7D) and a simple quadratic cost function (MSE) respectively,

![equation](https://latex.codecogs.com/gif.latex?L%5E%7B%5Cepsilon%7D_%7BPPO%7D%20%3D%20%5Cfrac%7B1%7D%7BM%7D%5Csum_%7Bt%2Ci%7Dmin%5Cleft%20%5B%20A%5Ei_t%5Cfrac%7B%5Cpi_%7Bnew%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%7B%5Cpi_%7Bold%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%2Cclip_%7B%5Cepsilon%7D%5Cleft%28%20A%5Ei_t%5Cfrac%7B%5Cpi_%7Bnew%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%7B%5Cpi_%7Bold%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%20%5Cright%20%29%20%5Cright%20%5D)

![equation](https://latex.codecogs.com/gif.latex?L_%7BMSE%7D%20%3D%20%5Cfrac%7B1%7D%7BMT%7D%5Csum_%7Bt%2Ci%7D%5Cleft%20%5C%7C%20V%28s_t%5Ei%29%20-%20%5Chat%7BR%7D_t%5Ei%20%5Cright%20%5C%7C%5E%7B2%7D)

where  ![equation](https://latex.codecogs.com/gif.latex?M%2C%20%5Cpi%28a%5Ei_t%7Cs%5Ei_t%29%2C%20V%28s_t%5Ei%29%20%2C%20%5Chat%7BR%7D_t%5Ei)  are the number of samples (=timepoints x agents), the policy, value function, and the discounted rewards respectively.

The Policy Network is a stochastic policy, meaning that, although the mean of the action is a deterministic function of the state, the action is selected randomly from a Gaussian distribution around this mean. It is important to note that it does not suffice to sample the action from a Gaussian to enhance exploration at the beginning and then decrease the noise gradually. This will result in an unstable learning procedure. Hence, the standard deviation of the Gaussian in the policy must also be learned via gradient descent. 
In addition, I chose a (truncated) Generalized Advantage Estimation with the discounting factor ![equation](https://latex.codecogs.com/gif.latex?%5Clambda),

![equation](https://latex.codecogs.com/gif.latex?A_t%20%3D%20%5Csum_%7Bi%3D0%7D%5E%7BT-%28t-1%29%7D%28%5Cgamma%5Clambda%29%5Ei%5Cdelta_%7Bt&plus;i%7D)

where ![equation](https://latex.codecogs.com/gif.latex?%5Cdelta_t%20%3D%20r_t%20&plus;%20%5Cgamma%20V%28s_%7Bt&plus;1%7D%29-%20V%28s_%7Bt%7D%29)

For more details, the reader is referred to the [PPO paper](https://arxiv.org/pdf/1707.06347.pdf).

### Neural Network Architectures
Both the actor and the critic neural networks have two layers with 256 nodes each. Both have ReLUs as activation functions and hence, the only difference is in their output. While the critic outputs a single linear node giving the state-value function estimation, the actor's output is 4-dimensional representing the torch in both joints. Furthermore, this output is passed through a hyperbolic tangent before it is fed as the mean to a Normal distribution, from which the final action is sampled. 

### Parameters

Parameter | episode | lr_act | lr_crit | discount | lmda| epsilon | eps. discount |  SGD_epoch | batch_size |
---|---|---|---|---|---|---|---|---|---|
Value | 200 | 2e-4 | 2e-4 | 0.99 | 0.95 | 0.2 | 0.999 | 6 | 64 |
Description | max. # episodes | learning rate actor | learning rate critic | reward discount factor | GAE factor | clipping threshold | discounting factor each episode |nr. grad. desc. epochs | mini-batch |

## Results

The agent took 172 episodes to be considered solved, although the +30 reward threshold was reached already at around 125 episodes and it was consistently above it afterward. Hence, the learning procedure was stable, see fig. "Mean Rewards". 

![Mean Rewards](https://github.com/hcruiz/Continuous_Control/blob/master/Mean_rewards.png "Mean Rewards")

The stability of the procedure can be appreciated when looking at the actor and critic losses, see fig. "AC losses". Here, the 'Episodes' denote the gradient descent epochs made over the entire training. On the left, we see that the policy/actor loss decreases rapidly at the beginning and stays relatively stable during training. On the right, we see that, in the beginning, the critic loss increases rapidly and then stabilizes around 0.1. This increase can be understood from the initial (very small) values of the discounted reward. 
An interesting observation is that if the nr. of gradient descent epochs (SGD_epoch) is increased, both losses increase and learning becomes unstable. This can be understood as an effect of overfitting early on the initial (non-stationary) reward distribution. The effect of this 'fixation' adds up when the reward distribution is shifted towards higher values and the errors accumulate, leading to a collapse in the mean reward.
![AC losses](https://github.com/hcruiz/Continuous_Control/blob/master/AC_Losses.png "AC losses")

## Future Work

Although the agent learned and attained the objective, I believe it can be more efficient and achieve higher rewards, if I can stabilize training for a higher number of gradient descent epochs. This can accelerate learning at the beginning while keeping the agent stable. 
There are several possible ways out:
* Optimize the values of the relevant hyper-parameters (although I already did this, it might be worth spending a bit more time on it): learning rates, SGD_epoch, epsilon.
* Normalize the returns and use batch norm (to maintain the distribution more 'stationary')
* Use a target network and perform soft updates on the network to smoothen out and forget earlier updates.

Also, if the losses do not decrease with the above changes, spending more time optimizing the network architecture could be useful since, from the critic loss, we see that error is still relatively high. 

Other interesting considerations are:
* Compare performance with [DDPG](https://arxiv.org/abs/1509.02971), [TRPO](https://arxiv.org/abs/1502.05477), [A3C](https://arxiv.org/pdf/1602.01783.pdf) or [D4PG](https://openreview.net/pdf?id=SyZipzbCb).

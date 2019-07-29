import time
from collections import deque
import progressbar as pb
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# Get device available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ",device)

def train(envs, policy, critic, episode, lr_act=2e-4, lr_crit=2e-4, discount=.99, lmda=0.95,
          epsilon=0.2, SGD_epoch=4, batch_size=64):
    
    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()
    
    optimizer = optim.Adam(policy.parameters(), lr=lr_act)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_crit)

    # keep track of progress
    mean_rewards = []
    scores_window = deque(maxlen=100)  # last 100 scores
    ploss = []
    closs = []

    for e in range(episode):

        # collect trajectories
        States, LogPs, Actions, Rewards, Vs, scores = get_samples(envs, policy, critic)
        # return estimation 
        returns_estimation = discounted_returns(Rewards, discount)
        #get advantage estimation
        advantage = advantage_estimate(Rewards, discount, Vs, lmda)
        #normalize advantage
        advantage = (advantage - np.mean(advantage))/(np.std(advantage)+1.e-10) 

        total_rewards = np.sum(Rewards, axis=0)

        #transform to tensors and flatten: States, LogPs, Actions, advantage,returns_estimation
        States = torch.tensor(States, dtype=torch.float, device=device).view(-1,States.shape[-1])
        returns_estimation = torch.tensor(returns_estimation, dtype=torch.float, device=device).view(-1)
        advantage = torch.tensor(advantage, dtype=torch.float, device=device).view(-1)
        Actions = torch.tensor(Actions, dtype=torch.float, device=device).view(-1,Actions.shape[-1])
        LogPs = torch.tensor(LogPs, dtype=torch.float, device=device).view(-1)

        # gradient ascent step
        for _ in range(SGD_epoch):
            
            for minibatch in get_minibatches(len(States), batch_size):
                states, log_ps, actions, adv = States[minibatch], LogPs[minibatch], Actions[minibatch], advantage[minibatch]
                returns = returns_estimation[minibatch]

                #PPO return function
                policy_loss = PPO(policy, states, log_ps, actions, adv, epsilon=epsilon)
                # Critic loss
                critic_loss = Lcritic(critic, states, returns)
                #update policy
                optimizer.zero_grad()
                policy_loss.backward()
#                 nn.utils.clip_grad_norm_(policy.parameters(), 5)
                optimizer.step()
                #update critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
#                 nn.utils.clip_grad_norm_(critic.parameters(), 5)
                critic_optimizer.step()
                
            policy_loss = PPO(policy, States, LogPs, Actions, advantage, epsilon=epsilon)
            critic_loss = Lcritic(critic, States, returns_estimation)  
            ploss.append(policy_loss.data)
            closs.append(critic_loss.data)

        # the clipping parameter reduces as time goes on
        epsilon*=.999

        # get the average reward of the parallel environments
        score = np.mean(total_rewards)
        mean_rewards.append(score)
        scores_window.append(score)       # save most recent score

        # display some progress every 5 iterations
        if (e+1)%20 == 0 :
            print("Episode: {0:d}, score (averaged over agents): {1:f}".format(e+1,np.mean(total_rewards)))
            print("Policy loss: {} | Critic loss: {}".format(policy_loss.data,critic_loss.data))
            print("Clipping threshold: {}".format(epsilon/0.999))
            torch.save(policy.state_dict(), 'Reacher-ckpt.policy')
            torch.save(critic.state_dict(), 'Reacher-ckpt.critic')
        # update progress widget bar
        timer.update(e+1)
        # check if environment is solved and save
        if np.mean(scores_window)>=30: 
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e+1-100, np.mean(scores_window)))
            # Save actor and critic
            torch.save(policy.state_dict(), 'Reacher.policy')
            torch.save(critic.state_dict(), 'Reacher.critic')
            print('Networks saved')
            break

    timer.finish()
    return mean_rewards, ploss, closs

def get_samples(env, policy, critic, train_mode=True):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=train_mode)[brain_name]     # reset the environment    
    # get the current state (for each agent)
    states = env_info.vector_observations 
    # number of agents
    num_agents = len(env_info.agents)
    # initialize containers
    scores = np.zeros(num_agents)                          
    States, LogPs, Actions, Rewards, Vs = [], [], [], [], []
    #loop until episode is done
    while True:
        states = torch.tensor(states,dtype=torch.float,device=device)
        log_Ps, actions = policy(states) # select an action (for each agent)
        actions = actions.cpu().detach().numpy()
        a = np.clip(actions, -1, 1) # all actions between -1 and 1
        env_info = env.step(a)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        s_vals = critic(states).cpu().detach().numpy()     # get state value
        #add info to episode lists
        States.append(states.cpu().detach().numpy())
        LogPs.append(log_Ps.cpu().detach().numpy())
        Actions.append(actions)
        Rewards.append(np.array(rewards))
        Vs.append(s_vals)
        #perform step
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    
    States = np.asarray(States)
    LogPs = np.asarray(LogPs)
    Actions = np.asarray(Actions)
    Rewards = np.asarray(Rewards)
    Vs = np.squeeze(np.asarray(Vs))
    return States, LogPs, Actions, Rewards, Vs, scores

def discounted_returns(rewards, discount):
    discouted_rewards =  np.asarray([rewards[i]*discount**i for i in range(len(rewards))],dtype=np.float32)
    #convert to future rewards
    future_rewards = np.cumsum(discouted_rewards[::-1],axis=0)[::-1]
    returns = future_rewards.copy()
    return returns

def advantage_estimate(rewards, discount, Vs, lmda):
    V_next = np.concatenate((Vs[1:],np.zeros(Vs[0].shape)[np.newaxis]),axis=0)
    TD_error = rewards + discount*V_next - Vs
    truncation_weights = [[(discount*lmda)**i] for i in range(len(rewards))]
    advantage = [np.sum(TD_error[i:]*truncation_weights[::-1][i:][::-1],axis=0) for i in range(len(rewards))]
    advantage = np.asarray(advantage)
    return advantage

def PPO(policy, states, log_ps, actions, advantage, epsilon=0.1, debug=False):
    if debug:
        print(states.shape, actions.shape)

    # convert states to policy
    new_logPs, _ = policy(states, actions)
    #compute probabilities ratio = new_probs/old_probs
    Delta = new_logPs - log_ps
    ratio = torch.exp(Delta)
    if debug:
        print(ratio)
        print(Delta)

    #compute returns and clipped function
    min_ratio = torch.min(advantage*ratio, advantage*torch.clamp(ratio, 1-epsilon, 1+epsilon))
    
    return -torch.mean(min_ratio)

def Lcritic(critic, states, returns_estimation):
    # convert states to value
    values = critic(states).view(-1)
    return torch.nn.functional.mse_loss(returns_estimation, values)

def get_minibatches(data_size, batch_size, suffle=True):
    indices = np.arange(data_size)
    if suffle:
        np.random.shuffle(indices)
    for mb in range(0,data_size-batch_size + 1, batch_size):
        get_indx = indices[mb:mb+batch_size]
        yield get_indx
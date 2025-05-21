#Run Code
#source .venv/bin/activate
#python src/lacunaBoard.py

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical

from player import Player


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indicies = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indicies)
        batches = [indicies[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    #called in Actor-Critic Networks to store data
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    #gets called to empty the buffer/meomery, gets called after learning
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

#policy network - maps states to action probabilities
#This is used in choose_action and during training for policy update
class PPOActorNetwork(nn.Module):#Policy
    def __init__(self, n_actions, input_dims, alpha, fc1_dims, fc2_dims, chkpt_dir):
        super(PPOActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions)#,
        )

        self.log_std = nn.Parameter(T.zeros(1, n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        #CONTINUOUS (GAUSIAN.)
        mean = self.actor(state) #get mean from the actor
        log_std = self.log_std.expand_as(mean) #Log standard deviation (use learnable parameters)
        std = T.exp(log_std)# Calculate standard deviation from log std
        return mean, std# Return mean and standard deviation for Gaussian distribution


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

#Value network - estimates how good a given state is
class PPOCriticNetwork(nn.Module):#Value
    def __init__(self, input_dims, alpha, fc1_dims, fc2_dims, chkpt_dir):
        super(PPOCriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        #Feedfoward neural netowrk for estimating the state-value function V(s)
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class PPOAgent(Player):
    def __init__(self, n_actions = 2, input_dims = (2,)):

        #change me for fine tunring
        self.gamma = 0.99 #
        self.policy_clip = 0.2 #policy rate of change max
        self.n_epochs = 100
        self.gae_lambda = 0.95
        self.critic_coeff = 0.5
        self.batch_size=64
        self.alpha = 0.003 #leanring rate

        self.fc1_dims=256 #number of neurons in the first hidden layer
        self.fc2_dims=256 #number of neurons in the second hidden layer

        self.chkpt_dir='tmp/ppo'

        self.observation = input_dims

        #create networks and memory
        self.actor = PPOActorNetwork(n_actions, input_dims, self.alpha, self.fc1_dims, self.fc2_dims, self.chkpt_dir)
        self.critic = PPOCriticNetwork(input_dims, self.alpha, self.fc1_dims, self.fc2_dims, self.chkpt_dir)
        self.memory = PPOMemory(self.batch_size)

    #----MEMORY FUNCTIONS----
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('----saving networks----')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('----loading networks----')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def select_action (self):
        state = T.tensor([self.observation], dtype=T.float).to(self.actor.device) #replace this with the states of the lacuna board

        mean, std = self.actor(state)
        dist = T.distributions.Normal(mean, std)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value


    def receive_observation(self, observation, reward, done, info):
        self.observation = observation
        store_memory()
        print(f"You got a reward of {reward:0.2f}, is the game done? {done}")
        if len(self.states) >= batch_size:
            self.learn()


    def plot_learning_curve(x, scores, figure_file):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(figure_file)  

    def save(self, filepath):
        print(f"SAVE ME!!!!!!!!! (Later)")

    #main leanring fubnction, call this to train model
    def learn(self):
        #get training data from memory
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr

            #Calculte Advantage using GAE(Gneneralised Advantage Estimation)
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t #advantage[t] = how much better the taken action was than expected
            
            #Convert to PyTorch tensor
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            #loop over minibatches
            for batch in batches:
                #sample batch
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                #get new policy and value estimations using networks
                dist = self.actor(states) #get new action distribution
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value) #flatten critic output

                #PPO Ratio and Clipping
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp() # this is the policy ratio equation

                #Clipped Surrogate loss to prevent big jumps in policy update
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean() #clipped surrogate loss equation

                #critic loss (MSE between returns and value estimates)
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                #Backproperate combined loss
                total_loss = actor_loss + self.critic_coeff*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        #clear memort after update
        self.memory.clear_memory()


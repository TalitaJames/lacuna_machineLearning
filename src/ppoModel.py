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
    def __init__(self, mini_batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.mini_batch_size = mini_batch_size

    def generate_batches(self):
            n_states = len(self.states)
            batch_start = np.arange(0, n_states, self.mini_batch_size)
            indicies = np.arange(n_states, dtype=np.int64)
            np.random.shuffle(indicies)
            batches = [indicies[i:i+self.mini_batch_size] for i in batch_start]

            return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        #print(f"I just stored a state with {action}, {probs}, {vals}, {reward}")
        self.states.append(state)
        self.actions.append(action.detach().cpu().numpy().reshape(-1))
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

    def __len__(self):
        return len(self.states)

#policy network - maps states to action probabilities
#This is used in choose_action and during training for policy update
class PPOActorNetwork(nn.Module):#Policy
    def __init__(self, n_actions, obs_dim, alpha, fc1_dims, fc2_dims, chkpt_dir):
        super(PPOActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(*obs_dim, fc1_dims),
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


#Value network - estimates how good a given state is
class PPOCriticNetwork(nn.Module):#Value
    def __init__(self, obs_dim, alpha, fc1_dims, fc2_dims, chkpt_dir):
        super(PPOCriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        #Feedfoward neural netowrk for estimating the state-value function V(s)
        self.critic = nn.Sequential(
                nn.Linear(*obs_dim, fc1_dims),
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

class PPOAgent(Player):
    def __init__(
        self,
        n_actions,          # Number of action dimensions (e.g., 2 for (x, y) moves)
        obs_dim,            # Number of dimensions in the observation vector

        gamma,              # Discount factor for future rewards
        policy_clip,        # Policy rate of change max
        n_epochs,           # Per learning update
        gae_lambda,         # Lambda for Generalized Advantage Estimation (GAE)
        critic_coeff,       # Weighting for critic loss in total loss
        alpha,              # Learning rate for both actor and critic

        batch_size,         # Number of transitions before learning
        mini_batch_size,    # Size of each training batch
        fc1_dims,           # Number of neurons in the first hidden layer
        fc2_dims,           # Number of neurons in the second hidden layer
        chkpt_dir,          # Directory to save/load model checkpoints
        ):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.critic_coeff = critic_coeff
        self.alpha = alpha
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir

        #create networks and memory
        self.actor = PPOActorNetwork(n_actions, (obs_dim,), self.alpha, self.fc1_dims, self.fc2_dims, self.chkpt_dir)
        self.critic = PPOCriticNetwork((obs_dim,), self.alpha, self.fc1_dims, self.fc2_dims, self.chkpt_dir)
        self.memory = PPOMemory(self.mini_batch_size)

        self.lastAction = None
        self.lastActionProbs = None
        self.lastActionVals = None

    #----MEMORY FUNCTIONS----
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def select_action (self):
        state = T.tensor([self.observation], dtype=T.float).to(self.actor.device)

        mean, std = self.actor(state)
        dist = T.distributions.Normal(mean, std)
        value = self.critic(state)
        action = dist.sample()
        self.lastAction = action

        log_probs = dist.log_prob(action).sum(dim=-1)  # sum over action dimensions
        probs = log_probs.item()
        action = action.cpu().numpy().reshape(-1)  # Always shape (n_actions,)


        value = value.item()
        self.lastActionProbs = probs
        self.lastActionVals = value
        return action


    def receive_observation(self, observation, reward, done, info):
        self.observation = observation

        if self.lastAction is not None:
            self.remember(observation, self.lastAction, self.lastActionProbs, self.lastActionVals, reward, done)

        if len(self.memory) >= self.batch_size:
            self.learn()


    def save(self, filepath):
        '''Save the actor and critic networks to a file.'''
        T.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, filepath)
        print(f"Saved PPO agent networks to {filepath}")

    def load(self, filepath):
        '''Load the actor and critic networks from a file.'''
        checkpoint = T.load(filepath, map_location=self.actor.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded PPO agent networks from {filepath}")


    #main leanring fubnction, call this to train model
    def learn(self):
        # get training data from memory
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr

            # Calculate Advantage using GAE (Generalised Advantage Estimation)
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            # Convert arrays to PyTorch tensors
            advantage = T.tensor(advantage, dtype=T.float32).to(self.actor.device)
            values = T.tensor(values, dtype=T.float32).to(self.actor.device)

            # Pre-convert all arrays to np.arrays for indexing
            state_arr = np.array(state_arr)
            action_arr = np.array(action_arr)
            old_prob_arr = np.array(old_prob_arr)

            # loop over minibatches
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float32).to(self.actor.device)

                # get new policy and value estimations using networks
                mean, std = self.actor(states)
                dist = T.distributions.Normal(mean, std)
                critic_value = self.critic(states).squeeze()

                # PPO Ratio and Clipping
                new_probs = dist.log_prob(actions).sum(dim=-1)  # sum over action dimensions
                prob_ratio = new_probs.exp() / old_probs.exp()

                # Clipped surrogate loss
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # backprop
                total_loss = actor_loss + self.critic_coeff * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


if __name__ == '__main__':
    print("OI YOU SNEEKY LITTLE CU*T YOU CANT RUN THE CODE FROM HERE! RUN IT FROM THE MAIN YOU SMALL SMOOTH BRAINED INDIVIDUAL!!")
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
                #nn.Softmax(dim=-1)<---DISCRETE
        )

        self.log_std = nn.Parameter(T.zeros(1, n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        #DISCRETE (SOFTMAX)
        # dist = self.actor(state)
        # dist = Categorical(dist)#The output is a catergorical distribution which allows sampling actions and computing log-probabilites
        # return dist

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
    def __init__(self, n_actions, input_dims):

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

        #create networks and memory
        self.actor = PPOActorNetwork(n_actions, input_dims, self.alpha, self.fc1_dims, self.fc2_dims, self.chkpt_dir)
        self.critic = PPOCriticNetwork(input_dims, self.alpha, self.fc1_dims, self.fc2_dims, self.chkpt_dir)
        self.memory = PPOMemory(self.batch_size)

    #----MEMORY FUNCTIONS----
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('----saving models----')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('----loading models----')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    #----TRAINING FUNCTIONS----
    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device) #replace this with the states of the lacuna board

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    #main leanring fubnction, call this to train model
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + self.critic_coeff*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        #clear once leanring is done
        self.memory.clear_memory()

    def plot_learning_curve(x, scores, figure_file):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(figure_file)

if __name__ == '__main__':
    print("HELLO WORLD :D")

    #place holder for lacunaboard env
    # env = gym.make('CartPole-v0')
    # N = 20
    # batch_size = 5
    # n_epochs = 4
    # alpha = 0.0003


    # ppoAgent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size,
    #                     alpha=alpha, n_epochs=n_epochs,
    #                     input_dims=env.observation_space.shape)

    # n_games = 300

    # best_score = env.reward_range[0]
    # score_history = []

    # learn_iters = 0
    # avg_score = 0
    # n_steps = 0

    # for i in range(n_games):
    #     observation = env.reset()
    #     done = False
    #     score = 0
    #     while not done:
    #         action, prob, val = agent.choose_action(observation)
    #         observation_, reward, done, info = env.step(action)
    #         n_steps += 1
    #         score += reward
    #         agent.remember(observation, action, prob, val, reward, done)
    #         if n_steps % N == 0:
    #             agent.learn()
    #             learn_iters += 1
    #         observation = observation_
    #     score_history.append(score)
    #     avg_score = np.mean(score_history[-100:])

    #     if avg_score > best_score:
    #         best_score = avg_score
    #         agent.save_models()

    #     print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,'time_steps', n_steps, 'learning_steps', learn_iters)

    # x = [i+1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)
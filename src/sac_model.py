import os
import numpy as np
import torch

import utils
from player import Player
from sac_actor import DiagGaussianActor
from sac_critic import DoubleQCritic
from sac_replayBuffer import SACReplayBuffer

# from github code
import torch.nn.functional as F

'''
Code adapted from:
https://arxiv.org/abs/1801.01290
'''

class SACAgent(Player):
    '''SAC player with code addapted from https://github.com/denisyarats/pytorch_sac'''
    def __init__(
        self,
        obs_dim,                   # Dimension of observation/state space
        action_dim,                # Dimension of action space
        action_range,              # Tuple (min, max) for action values (for all action)
        device,                    # what architecture to run, 'cpu' or 'cuda'
        discount=0.99,             # Discount factor for future rewards
        init_temperature=0.1,      # Initial entropy temperature (alpha)
        alpha_lr=3e-4,             # Learning rate for alpha (entropy)
        alpha_betas=(0.9, 0.999),  # Adam betas for alpha optimizer

        actor_lr=3e-4,             # Learning rate for actor
        actor_betas=(0.9, 0.999),  # Adam betas for actor optimizer
        actor_update_frequency=1,  # How often to update actor (in steps)

        critic_cfg=None,           # Critic configuration
        critic_lr=3e-4,            # Learning rate for critic
        critic_betas=(0.9, 0.999), # Adam betas for critic optimizer
        critic_tau=0.005,          # Target smoothing coefficient (Polyak)
        critic_target_update_frequency=2, # How often to update target critic

        # Memory parameters
        batch_size=256,            # Batch size for updates
        replay_buffer_capacity=5000, # Capacity of the replay buffer
        learnable_temperature=True # Whether alpha is learnable
    ):
        # save parameters
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.step = 0 # number of actions/steps taken

        # buffer and previous state paramaters
        self.replay_buffer = SACReplayBuffer(
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
            capacity=replay_buffer_capacity,
            device=self.device
        )
        self.last_observation = None
        self.last_action = None
        self.last_reward = None
        self.last_done = None

        # Instantiate critic and target critic
        # obs_dim, action_dim, hidden_dim, hidden_depth):
        self.critic = DoubleQCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256, #TODO pick
            hidden_depth=2,
        ).to(self.device)
        self.critic_target = DoubleQCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256, #TODO pick
            hidden_depth=2,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Instantiate actor
        self.actor = DiagGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256, #TODO pick
            hidden_depth=2,
            log_std_bounds= (-20, 2) #TODO what is this for?
        ).to(self.device)

        # Entropy temperature (alpha, explore vs exploit)
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim  # Target entropy for exploration

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)

        self.train()
        self.critic_target.train()


    def receive_observation(self, observation, reward, done, info):
        # Store the latest transition (if not the first step)
        if self.last_observation is not None and self.last_action is not None:
            self.replay_buffer.add(
                self.last_observation, self.last_action, self.last_reward,
                observation, done)
        # Update internal state
        self.last_observation = observation
        self.last_reward = reward
        self.last_done = done

        self.step += 1
        # Optionally, trigger learning here
        if len(self.replay_buffer) >= self.batch_size:
            # print(f"Learning step, {self.step=}")
            self.update(self.step)


    def train(self, training=True):
        ''' set the neural nets (extended classes) to training mode'''
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self): # alpha is the exploration / exploitation param
        return self.log_alpha.exp()

    def select_action(self):
        # Use the most recent observation to select an action
        if self.last_observation is None:
            raise ValueError("No observation received yet!")
        action = self.act(self.last_observation, sample=True)
        self.last_action = action
        return action

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device) #
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.rsample() if sample else dist.mean
        action = torch.tanh(action)
        action = action.clamp(*self.action_range)
        return utils.to_np(action[0])

    def save(self, filepath):
        '''Save the actor, critic, and critic_target networks to a file.'''
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu().numpy(),
        }, filepath)
        print(f"Saved SAC agent networks to {filepath}")

    def load(self, filepath):
        '''Load the actor, critic, and critic_target networks from a file.'''
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.log_alpha = torch.tensor(checkpoint['log_alpha']).to(self.device)

        print(f"Loaded SAC agent networks from {filepath}")


    def update_critic(self, obs, action, reward, next_obs, not_done, step):# logger, step):
        dist = self.actor(next_obs) # get the
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        # logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, step): #logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # logger.log('train_actor/loss', actor_loss, step)
        # logger.log('train_actor/target_entropy', self.target_entropy, step)
        # logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            # logger.log('train_alpha/loss', alpha_loss, step)
            # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, step, logger=None):
        obs, action, reward, next_obs, not_done = self.replay_buffer.sample(
            self.batch_size)

        # logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, step) #logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)# logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)



if __name__ == "__main__":

    sacParams = utils.load_config('config/sac.json')
    print(sacParams)

    sac_agent = SACAgent(**sacParams)
    print(f"SAC agent created with name: {sac_agent}")

    # Fake testing data
    for i in range(20):
        obs = np.random.randn(sacParams["obs_dim"])
        reward = np.random.randn(1)
        done = False

        sac_agent.receive_observation(obs, reward, done, None)
        action = sac_agent.select_action()
        print(f"{i}) {action=}")


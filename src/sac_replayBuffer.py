import torch
import numpy as np

'''
Code adapted from: https://github.com/denisyarats/pytorch_sac
'''
class SACReplayBuffer():
    '''Buffer to store environment transitions.'''
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0 # index of next thing to be added
        self.last_save = 0
        self.full = False

    def __len__(self):
        '''Get the number of transitions in the buffer.'''
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        '''Add a transition to the buffer.
        Params:
        - obs: the observation of the current state
        - action: the action taken
        - reward: the reward received
        - next_obs: the observation of the next state
        - done: whether the episode has ended

        ie given obs, take action, recive reward and end up in next_obs
        '''
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        '''Sample a random batch_size of transitions from the buffer.'''
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones


if __name__ == "__main__":
    print("---- This module is not meant to be run directly. ----")

    buffer = SACReplayBuffer(obs_shape=(185,), action_shape=(2,), capacity=10, device='cpu')

    for _ in range(13):
        radom_obs = np.random.randn(185)
        random_action = np.random.randn(2)
        random_reward = np.random.randn(1)
        random_next_obs = np.random.randn(185)
        buffer.add(radom_obs, random_action, random_reward, random_next_obs, False)

    batch = buffer.sample(4)
    print(f"buffer size: {len(buffer)}, the batch shape: {batch[0].shape}")
    print([b.shape for b in batch])
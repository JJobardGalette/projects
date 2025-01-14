from collections import namedtuple
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from maxminQ.Qfunction import QNeural
from maxminQ.buffer import ExperienceReplayBuffer
from maxminQ.utils import running_average

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class AgentMaxMin:
    def __init__(self, agent_number, env, learning_rate=0, 
                 buffer_len=0, discount_factor=0.99, update_subset_size=1, batch_size=64, 
                 episode_number=1000, n_ep_running_average=50, early_stop=float('inf'),
                 epsilon_inf=0.01, epsilon_sup=1, epsilon_stop_decreasing=500, epsilon_method='linear'):
        self.learning_rate = learning_rate
        self.buffer_len = buffer_len
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.update_subset_size = update_subset_size
        self.episode_number = episode_number
        self.early_stop = early_stop
        
        self.env = env
        self.current_state = self.env.reset()[0]
        self.agent_number = agent_number
        self.agents = [QNeural(len(self.env.observation_space.high), self.env.action_space.n) for _ in range(self.agent_number)]
        self.buffer = ExperienceReplayBuffer(self.buffer_len)
        self.optimizers = [optim.Adam(self.agents[i].parameters, lr=self.learning_rate) for i in range(self.agent_number)]

        self.done_or_truncated = False
        self.current_episode_reward = 0
        self.Qmin = None
        self.compute_Q_min
        
        self.EPISODES = trange(self.episode_number, desc='Episode: ', leave=True)
        self.episode_list_reward = []
        self.episode_number_of_steps = []
        self.n_ep_running_average = n_ep_running_average

        self.epsilon_inf = epsilon_inf
        self.epsilon_sup = epsilon_sup
        self.epsilon_stop_decreasing = epsilon_stop_decreasing
        self.epsilon_method = epsilon_method
        self.epsilon = epsilon_sup

    def take_step(self, action):
        if not self.done_or_truncated:
            next_state, reward, done, truncated = self.env.step(action)
            self.done_or_truncated = done or truncated
            self.state = next_state
            self.current_episode_reward += reward
            self.buffer.append(Experience(self.state, action, reward, next_state, done))
        
    def compute_Q_min(self, states):
        tensor_min = torch.zeros((self.agent_number, states.shape[0], self.env.action_space.n))
        for i in range(self.agent_number):
            tensor_min[i] = self.agents[i](self.current_state)
        self.Qmin = tensor_min.min(1).squeeze()

    
    def update_network(self):
        """Training funtion:
            1. sample from buffer
            2. Compute Q functions
            3. Find the minimum
            4. update the network(s)"""
        states, actions, rewards, next_states, dones = self.buffer.sample(self.buffer_len)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).detach()
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        next_q_values = self.Qmin(next_states).max(1).detach()
        targets = rewards + self.discount_factor*next_q_values*(1-dones)
        index_to_update = np.random.sample(range(len(self.agents)), self.update_subset_size)
        for i in index_to_update:
            q_values = self.agents[i](states).gather(dim=1, index=actions).squeeze
            loss = nn.functional.mse_loss(q_values, targets)
            self.optimizers[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agents[i].parameters(), max_norm=1.)
            self.optimizers[i].step()

    
    def train(self):
        for i in range(self.episode_number):
            self.current_state = self.env.reset()[0]
            self.done_or_truncated = False
            self.current_episode_reward = 0
            self.epsilon = self.compute_epsilon(i)
            while not self.done_or_truncated:
                self.take_step()
                t+=1
                self.train()
            self.episode_list_reward.append(self.current_episode_reward)
            self.episode_number_of_steps.append(t)

            self.EPISODES.set_description(
                "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - Epsilon: {:.2f}".format(
                i, self.current_episode_reward, t, running_average(self.episode_list_reward, self.n_ep_running_average)[-1],
                running_average(self.episode_number_of_steps, self.n_ep_running_average)[-1], self.epsilon))
            if running_average(self.episode_list_reward, self.n_ep_running_average) > self.early_stop:
                break
        self.env.close()
        self.output_results()
    
    def select_action(self, state):
        if np.random.random() > self.epsilon:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            return self.Qmin(state_tensor).max()[1].item()
        else:
            return self.env.action_space.sample()
    
    def output_results(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        ax[0].plot([i for i in range(1, len(self.episode_reward_list)+1)], self.episode_reward_list, label='Episode reward')
        ax[0].plot([i for i in range(1, len(running_average(
            self.episode_reward_list, self.n_ep_running_average))+1)], running_average(
            self.episode_reward_list, self.n_ep_running_average), label='Avg. episode reward')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Total reward')
        ax[0].set_title('Total Reward vs Episodes')
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        ax[1].plot([i for i in range(1, len(self.episode_number_of_steps)+1)], self.episode_number_of_steps, label='Steps per episode')
        ax[1].plot([i for i in range(1, len(running_average(
            self.episode_number_of_steps, self.n_ep_running_average))+1)], running_average(
            self.episode_number_of_steps, self.n_ep_running_average), label='Avg. number of steps per episode')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Total number of steps')
        ax[1].set_title('Total number of steps vs Episodes')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        plt.show()

    def compute_epsilon(self, i):
        if self.epsilon_method == 'linear':
            return max(self.epsilon_inf, self.epsilon_sup-(self.epsilon_sup-self.epsilon_inf)*(i-1)/(self.epsilon_stop_decreasing-1)) # linear
        elif self.epsilon_method == 'exponential':
            return max(self.epsilon_inf, self.epsilon_sup*(self.epsilon_inf/self.epsilon_sup)**((i)/self.epsilon_stop_decreasing)) # exponential
        else:
            raise ValueError('method is not supported ("linear" or "exponential")')
        


    
if __name__ == '__main__':
    env = gymnasium.make('MountainCar-v0')
    maxmin_agent = AgentMaxMin(env)
    maxmin_agent.train()
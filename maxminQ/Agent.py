from collections import namedtuple
import gymnasium
import torch

from maxminQ.Qfunction import QNeural
from maxminQ.buffer import ExperienceReplayBuffer
env = gymnasium.make('MountainCar-v0')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class AgentMaxMin:
    def __init__(self, agent_number, env, learning_rate=0, buffer_len=0, discount_factor=0.99):
        self.learning_rate = learning_rate
        self.buffer_len = buffer_len
        self.discount_factor = discount_factor
        self.batch_size = 64
        
        self.env = env
        self.current_state = self.env.reset()[0]
        self.agent_number = agent_number
        self.learners = [QNeural(len(self.env.observation_space.high), self.env.action_space.n) for _ in range(self.agent_number)]
        self.buffer = ExperienceReplayBuffer(self.buffer_len)

        self.done_or_truncated = False
        self.current_episode_reward = 0

    def take_step(self, action):
        if not self.done_or_truncated:
            next_state, reward, done, truncated = self.env.step(action)
            self.done_or_truncated = done or truncated
            self.state = next_state
            self.current_episode_reward += reward
            self.buffer.append(Experience(self.state, action, reward, next_state, done))
    
    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.buffer_len)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        

    

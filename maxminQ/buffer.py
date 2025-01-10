from collections import deque

import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)
        self.env.reset()
    def sample(self, n):
        """sample n element from the buffer"""
        if n > len(self.buffer):
            raise IndexError('Sample size exceed buffer length')
        else:
            indices = np.random.choice(len(self.buffer), size=n, replace=False)
            batch = [self.buffer[i] for i in indices]
            return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)
    
    def __append__(self, x):
        self.buffer.append(x)

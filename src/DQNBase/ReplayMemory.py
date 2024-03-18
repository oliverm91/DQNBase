from collections import deque
import random

from DQNBase.Experience import Experience


class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.memory.append(experience)

    def sample(self, batch_size: int) -> deque[Experience]:
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
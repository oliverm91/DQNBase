from abc import ABC

import numpy as np
import torch


class Enviroment(ABC):
    def __init__(self, state: np.ndarray, n_actions: int, device: str) -> None:
        self.state = state
        self.action_space = [i for i in range(n_actions)]
        self.device = device
        self.done = False

    def reset(self):
        self.state = np.zeros(self.state.shape)

    def n_actions_available(self) -> int:
        return len(self.action_space)
    
    def take_action(self, action: int):
        reward, self.done = self.step(action)
        return torch.tensor([reward], device=self.device)
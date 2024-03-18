import random
import torch
from EpsilonGreedyStrategy import EpsilonGreedyStrategy
from DQN import DQN

class Agent:
    def __init__(self, strategy: EpsilonGreedyStrategy, n_actions: int, device: str) -> None:
        self.current_step = 0
        self.strategy, self.n_actions, self.device = strategy, n_actions, device

    def select_action(self, state, policy_net: DQN) -> int:
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.n_actions) # explore
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device) # exploit

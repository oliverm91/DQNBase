import math


class EpsilonGreedyStrategy:
    def __init__(self, start: float, end: float, decay: float) -> None:
        if start > 1:
            raise ValueError(f'start value of an Epsilon Greedy Strategy can not be greater than 1. Value received: {start}')
        if end <= 0:
            raise ValueError(f'end value of an Epsilon Greedy Strategy can smaller or equal to 0. Value received: {end}')
        if not 0<decay<1:
            raise ValueError(f'decay must be between 0 and 1 excluded. Value received: {decay}')
        
        self.start, self.end, self.decay = start, end, decay

    def get_exploration_rate(self, current_step: int) -> None:
        return self.end + (self.start - self.end) * math.exp(-current_step * self.decay)
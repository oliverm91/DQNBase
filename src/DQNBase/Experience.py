class Experience:
    __slots__ = 'state', 'action', 'next_state', 'reward'
    def __init__(self, state, action, next_state, reward):
        self.state, self.action, self.next_state, self.reward = state, action, next_state, reward
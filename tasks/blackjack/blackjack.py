from collections import Counter, defaultdict
import numpy as np


class StreamingQTabulator:
    def __init__(self, max_size=10000, gamma=1.):
        """Initialize the episode """
        self.max_size = max_size
        self.time_step = 0
        self.gamma = gamma

        # Keep track of return
        self.time_step_rewards = np.zeros(max_size)

        # Keep track of which state/actions occured at which step
        self.visits = defaultdict(lambda: defaultdict(lambda: {'time_steps': []}))

    def iter_states(self):
        for i in self.visits:
            yield i

    def iter_actions(self, state):
        for i in self.visits[state]:
            yield i

    def step(self, state, action, reward):
        """Take a step in the episode"""
        # Handle the state-action
        self.visits[state][action]['time_steps'].append(self.time_step)

        # Handle the reward
        self.time_step += 1
        if self.time_step == self.max_size:
            self._handle_split()

        # Apply discounted return
        # TODO: Fix efficiency of this call
        discounted_reward = np.array([reward * self.gamma ** i for i in range(self.time_step)])
        self.time_step_rewards[0:self.time_step] += discounted_reward

    def _handle_split(self):
        """Handle when an episode goes on longer than expected """
        raise NotImplementedError('Handle split is not implemented')

    def get_time_step_return(self, time_step):
        """Get the reward at a time step"""
        return self.time_step_rewards[time_step]

    def get_state_action_visits(self, s, a):
        """Get the time steps at which the state,action was visited """
        return self.visits[s][a]


# Test

e1 = [((19, 7, False), 0, 1.0)]
e2 = [((17, 1, False), 0, -1.0)]
e3 = [((13, 10, True), 1, 0), ((16, 10, True), 1, 0), ((16, 10, False), 1, 0), ((20, 10, False), 0, 1.0)]

t1 = StreamingQTabulator()
t2 = StreamingQTabulator()
t3 = StreamingQTabulator()
[t1.step(s, a, r) for s, a, r in e1]
[t2.step(s, a, r) for s, a, r in e2]
[t3.step(s, a, r) for s, a, r in e3]

assert t1.get_time_step_return(0) == 1.
assert t2.get_time_step_return(0) == -1

assert t3.get_time_step_return(0) == 1
assert t3.get_time_step_return(1) == 1
assert t3.get_time_step_return(2) == 1
assert t3.get_time_step_return(3) == 1.


N = 500
test_episode = [('x{}'.format(i), None, i) for i in range(N)]
t_test = StreamingQTabulator()
[t_test.step(s, a, r) for s, a, r in test_episode]
def sum_first_n(n):
    return n*(n+1) / 2
assert t_test.get_time_step_return(0) == sum_first_n(N-1)

a0 = t_test.get_time_step_return(int(N/2))
a1 = sum_first_n(N-1) - sum_first_n(int(N/2) - 1)
assert a0 == a1, "{},{}".format(a0, a1)

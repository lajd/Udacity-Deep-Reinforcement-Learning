from typing import List
import torch
import numpy as np
from tools.rl_constants import Experience
import itertools
from tools.misc import set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trajectories:
    """Store episode trajectories for PPO and related algorithms"""

    def __init__(self, seed):
        """"""
        set_seed(seed)
        self.memory = []

    def add(self, experience_list: List[Experience]):
        """Add new trajectory to memory."""
        self.memory.extend(experience_list)

    def sample(self, bsize: int):
        """Randomly sample a batch of experiences from memory."""

        def flatten(x):
            x = list(itertools.filterfalse(lambda i: i is None, x))
            if len(x) == 0:
                return None
            else:
                x = [i.reshape(1, -1) for i in x]
                return torch.cat(x, dim=0)

        states, actions, action_log_probs, returns, advantages, joint_state, joint_actions = map(
            lambda memory_tuple: flatten(memory_tuple), zip(*self.memory)
        )

        # Get advantage estimate
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Sample minibatches
        indices = torch.arange(len(states)).to(device)
        states = states.to(device)
        actions = actions.to(device)
        action_log_probs = action_log_probs.to(device)
        returns = returns.to(device)
        advantages = advantages.to(device)

        np.random.shuffle(indices)

        trajectory_batch = []

        for batch_idx in range(len(indices) // bsize):
            minibatch_indices = indices[bsize * batch_idx: bsize * (batch_idx + 1)]
            trajectory_batch.append((
                states[minibatch_indices],
                actions[minibatch_indices],
                action_log_probs[minibatch_indices],
                returns[minibatch_indices],
                advantages[minibatch_indices],
                joint_state[minibatch_indices] if joint_state is not None else [None] * len(minibatch_indices),
                joint_actions[minibatch_indices] if joint_actions is not None else [None] * len(minibatch_indices)
            ))

        return trajectory_batch

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)

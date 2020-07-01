import torch
from collections import namedtuple
from typing import List, Union, Optional
import numpy as np


def ensure_tensors(*args):
    outp = []
    for a in args:
        if a is None:
            outp.append(a)
        elif not isinstance(a, torch.Tensor):
            if isinstance(a, np.ndarray):
                outp.append(torch.from_numpy(a))
            elif isinstance(a, bool) or isinstance(a, int):
                outp.append(torch.LongTensor([a]))
            elif isinstance(a, float):
                outp.append(torch.FloatTensor([a]))
            else:
                raise ValueError("Unexpected type {}".format(type(a)))
        else:
            outp.append(a)
    return outp


class Experience:
    def __init__(self, state: torch.Tensor, action: torch.Tensor, reward: float, done: torch.Tensor, t_step: int, next_state: torch.Tensor=None):
        state, action, done, next_state = ensure_tensors(state, action, done, next_state)
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.t_step = t_step
        self.next_state = next_state

    def cuda(self):
        self.state = self.state.cuda()
        self.action = self.action.cuda()
        if self.next_state:
            self.next_state = self.next_state.cuda()
        return self

    def cpu(self):
        self.state = self.state.cpu()
        self.action = self.action.cpu()
        if self.next_state is not None:
            self.next_state = self.next_state.cpu()
        return self


class ExperienceBatch:
    def __init__(self, states: torch.Tensor, actions: torch.Tensor,
                 rewards: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor,
                 sample_idxs: Optional[torch.Tensor] = None, memory_streams: Optional[List[str]] = None,
                 is_weights: Optional[torch.FloatTensor] = None):

        states, actions, rewards, dones, next_states, sample_idxs, is_weights = ensure_tensors(
            states, actions, rewards, dones, next_states, sample_idxs, is_weights
        )
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.next_states = next_states
        self.sample_idxs = sample_idxs
        self.memory_streams = memory_streams
        self.is_weights = is_weights

    def to(self, device):
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.next_states = self.next_states.to(device)
        if self.is_weights is not None:
            self.is_weights = self.is_weights.to(device)
        return self

    def shuffle(self):
        # Add random permute
        r = torch.randperm(self.states.shape[0])
        self.memory_streams = [self.memory_streams[i] for i in r.tolist()]
        self.states = self.states[r]
        self.actions = self.actions[r]
        self.rewards = self.rewards[r]
        self.next_states = self.next_states[r]
        self.dones = self.dones[r]
        self.sample_idxs = self.sample_idxs[r]
        if self.is_weights is not None:
            self.is_weights = self.is_weights[r]

    def get_norm_is_weights(self):
        if self.is_weights is None:
            raise ValueError("IS Weights are undefined")
        return self.is_weights / self.is_weights.max()

    def __len__(self):
        return len(self.states)


class Action:
    def __init__(self, value: Union[int, float, list, np.ndarray], distribution: Optional[np.array] = None):
        self.value = value
        self.distribution = distribution


class Trajectories:
    def __init__(self, policy_outputs: Union[list, np.ndarray], states: Union[list, np.ndarray], actions: Union[list, np.ndarray], rewards: Union[list, np.ndarray]):
        self.policy_outputs = policy_outputs
        self.states = states
        self.actions = actions
        self.rewards = rewards


Environment = namedtuple("Environment", field_names=["next_state", "reward", "done"])

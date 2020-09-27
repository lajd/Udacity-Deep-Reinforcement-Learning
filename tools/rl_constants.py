import torch
from collections import namedtuple, OrderedDict
from typing import List, Union, Optional, Dict, Callable, Tuple
import numpy as np
from tools.scores import Scores


def ensure_tensors(*args):
    outp = []
    for a in args:
        if a is None:
            outp.append(a)
        elif not isinstance(a, torch.Tensor):
            if isinstance(a, np.ndarray):
                outp.append(torch.from_numpy(a))
            elif isinstance(a, (bool, np.bool)) or isinstance(a, (int, np.int, np.int64)):
                outp.append(torch.LongTensor([a]))
            elif isinstance(a, (float, np.float, np.float32, np.float64)):
                outp.append(torch.FloatTensor([a]))
            elif isinstance(a, (tuple, List)):
                # input is a tuple; ensure all are tensors
                temp_outp = torch.cat(ensure_tensors(*a), dim=0)
                outp.append(temp_outp)
            else:
                raise ValueError("Unexpected type {} -- {}".format(type(a), a))
        else:
            outp.append(a)
    return outp


class Action:
    def __init__(self, value, **kwargs):
        self.__dict__.update(kwargs)
        self.value = value


def concatenate_action_attributes(actions_list: Union[List[Action], Action], attribute_name: str, cat_dim=0):
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    if isinstance(actions_list, Action):
        return to_tensor(getattr(actions_list, attribute_name))
    else:
        tensor_list = [to_tensor(getattr(a, attribute_name)) for a in actions_list]
        return torch.cat(tensor_list, cat_dim)


class Experience:
    def __init__(self, state: torch.Tensor, action: Action, done: Optional[torch.Tensor] = None, reward: Optional[float] = None,
                 t_step: Optional[int] = None, next_state: Optional[torch.Tensor] = None,
                 joint_state: Optional[torch.Tensor] = None, joint_action: Optional[torch.Tensor] = None,
                 joint_next_state: Optional[torch.Tensor] = None, brain_name=None, agent_number=None, **kwargs):

        for name, value in kwargs.items():
            value = ensure_tensors(value)
            self.__dict__[name] = value

        state, done, next_state, joint_state, joint_action, reward = ensure_tensors(
            state, done, next_state, joint_state, joint_action, reward
        )
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.t_step = t_step
        self.next_state = next_state

        self.joint_state = joint_state
        self.joint_action = joint_action
        self.joint_next_state = joint_next_state

        self.brain_name = brain_name
        self.agent_number = agent_number

    def _get_tensor_attributes(self):
        return {k: v for k, v in self.__dict__.items() if (not callable(v) and not k.startswith('_') and isinstance(v, torch.Tensor))}

    def to(self, device: str):
        for k, v in self._get_tensor_attributes().items():
            setattr(self, k, v.to(device))
        return self

    def cpu(self):
        for k, v in self._get_tensor_attributes().items():
            setattr(self, k, v.cpu())
        return self


class ExperienceBatch:
    def __init__(self, states: torch.Tensor, actions: torch.Tensor,
                 rewards: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor,
                 sample_idxs: Optional[torch.Tensor] = None, memory_streams: Optional[List[str]] = None,
                 is_weights: Optional[torch.FloatTensor] = None, joint_states: Optional[torch.FloatTensor] = None,
                 joint_actions: Optional[torch.Tensor] = None, joint_next_states: Optional[torch.Tensor] = None, agent_num=None):

        states, actions, rewards, dones, next_states, sample_idxs, is_weights, joint_states, joint_actions = ensure_tensors(
            states, actions, rewards, dones, next_states, sample_idxs, is_weights, joint_states, joint_actions
        )
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.next_states = next_states
        self.sample_idxs = sample_idxs
        self.memory_streams = memory_streams
        self.is_weights = is_weights
        self.joint_states = joint_states
        self.joint_actions = joint_actions
        self.joint_next_states = joint_next_states
        self.agent_num = agent_num

    def _get_tensor_attributes(self):
        return {k: v for k, v in self.__dict__.items() if (not callable(v) and not k.startswith('_') and isinstance(v, torch.Tensor))}

    def to(self, device: torch.device):
        for k, v in self._get_tensor_attributes().items():
            setattr(self, k, v.to(device))
        return self

    def shuffle(self):
        # Add random permute
        r = torch.randperm(self.states.shape[0])
        self.memory_streams = [self.memory_streams[i] for i in r.tolist()]

        for k, v in self._get_tensor_attributes():
            setattr(self, k, v[r])

    def get_norm_is_weights(self):
        if self.is_weights is None:
            raise ValueError("IS Weights are undefined")
        return self.is_weights / self.is_weights.max()

    def __len__(self):
        return len(self.states)


Environment = namedtuple("Environment", field_names=["next_state", "reward", "done"])


class Brain:
    def __init__(self, brain_name: str, action_size: int, state_shape: int, observation_type: str, agents,
                 preprocess_state_fn: Callable = lambda state: state, preprocess_actions_fn: Callable = lambda action: action):

        self.brain_name = brain_name
        self.action_size = action_size
        self.state_shape = state_shape
        self.observation_type = observation_type
        self.agents = agents
        self.agent_scores = [Scores() for _ in range(len(agents))]

        self.preprocess_state_fn = preprocess_state_fn
        self.preprocess_actions_fn = preprocess_actions_fn

    def get_action(self, state: np.ndarray, joint_state: np.ndarray) -> Dict[str, List[Action]]:
        # select actions and send to environment
        r = []
        if len(self.agents) == 1:
            r.append(self.agents[0].get_action(state, joint_state=joint_state))
        else:
            assert len(self.agents) == len(state),\
                "Need the same number of agents as provided in the" \
                " state; found {} and {} respectively".format(len(self.agents), len(state))
            for a, s in zip(self.agents, state):
                s = s.unsqueeze(0)
                action = a.get_action(s, joint_state=joint_state)
                r.append(action)
        return {self.brain_name: r}

    def get_random_action(self, state: np.ndarray, joint_state: np.ndarray) -> Dict[str, List[Action]]:
        # select actions and send to environment
        r = []
        if len(self.agents) == 1:
            r.append(self.agents[0].get_random_action(state, joint_state=joint_state))
        else:
            assert len(self.agents) == len(state),\
                "Need the same number of agents as provided in the" \
                " state; found {} and {} respectively".format(len(self.agents), len(state))
            for a, s in zip(self.agents, state):
                r.append(a.get_random_action(s, joint_state=joint_state))
        return {self.brain_name: r}


class BrainSet:
    def __init__(self, brains: List[Brain]):
        self.brain_map = OrderedDict([(brain.brain_name, brain) for brain in brains])

    def get_actions(self, brain_states) -> Dict[str, List[Action]]:
        # Get the joint states/actions
        joint_brain_states = torch.cat([brain_states[brain_name] for brain_name in brain_states]).view(1, -1)

        new_brain_actions = {}
        for brain_name, state in brain_states.items():
            brain_action_map = self.brain_map[brain_name].get_action(state=state, joint_state=joint_brain_states)
            new_brain_actions.update(brain_action_map)
        return new_brain_actions

    def get_random_actions(self, brain_states) -> Dict[str, List[Action]]:
        brain_actions = {}
        joint_brain_states = torch.cat([brain_states[brain_name] for brain_name in brain_states]).view(1, -1)
        for brain_name, state in brain_states.items():
            brain_actions.update(self.brain_map[brain_name].get_random_action(state,  joint_state=joint_brain_states))
        return brain_actions

    def step_agents(self, next_brain_environment: dict, t: int):
        pass

    def brains(self):
        for b in self.brain_map.values():
            yield b

    def names(self):
        for n in self.brain_map:
            yield n

    def __getitem__(self, brain_name: str):
        return self.brain_map[brain_name]

    def __iter__(self):
        for brain_name, brain in self.brain_map.items():
            yield brain_name, brain


class RandomBrainAction:
    def __init__(self, action_dim: int, num_agents: int, continuous_actions: bool = True,
                 continuous_action_range: Tuple[float, float] = (-1, 1),
                 discrete_action_range: Tuple[int, int] = (0, 1)):
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.continuous_actions = continuous_actions
        self.continuous_action_range = continuous_action_range
        self.discrete_action_range = discrete_action_range

    def sample(self) -> np.ndarray:
        if self.continuous_actions:
            # uniform_distribution = torch.distributions.uniform.Uniform(*self.continuous_action_range)
            # sample = uniform_distribution.sample((self.num_agents, self.action_dim)).cpu().numpy()
            sample = np.random.uniform(self.continuous_action_range[0], self.continuous_action_range[1], (self.num_agents, self.action_dim)).astype(np.float32)
            return sample
        else:
            return np.random.random_integers(
                self.discrete_action_range[0],
                self.discrete_action_range[1] - 1,
                (self.num_agents, self.action_dim)
            )
            # # Assume discrete actions are all zeros with a single 1
            # sample = np.zeros((self.num_agents, self.action_dim))
            # unit_dimensions = np.random.random_integers(0, self.action_dim - 1, self.num_agents)
            # for i, j in enumerate(unit_dimensions):
            #     sample[i][j] = 1
            # return sample

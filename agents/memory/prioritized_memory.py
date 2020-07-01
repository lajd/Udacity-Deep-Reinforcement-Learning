import numpy as np
from collections import namedtuple, deque
import torch
from typing import Type, NamedTuple, Union, Optional, Callable, Dict
from tools.data_structures.sumtree import SumTree
from tools.rl_constants import Experience, ExperienceBatch
from tools.parameter_decay import ParameterScheduler
from tools.misc import set_seed
from itertools import islice
from torch.autograd import Variable
from typing import List
import random
from collections import Counter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, state_shape: tuple, capacity: int):
        self.capacity = capacity
        self.buffer = deque([
            Experience(
                state=torch.zeros(state_shape),
                action=torch.FloatTensor([0]),
                reward=0,
                next_state=None,
                done=torch.LongTensor([0]),
                t_step=-1
            ) for _ in range(capacity)
        ], maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __setitem__(self, key, value):
        self.buffer[key] = value

    def __getitem__(self, item):
        if isinstance(item, slice):
            return list(islice(self.buffer, item.start, item.stop, item.step or 1))
        else:
            return self.buffer[item]


class PrioritizedMemory:
    """ Memory buffer for storing and sampling experience

    Adapted from https://adventuresinmachinelearning.com/sumtree-introduction-python/
    """
    def __init__(self, capacity: int, state_shape: tuple, beta_scheduler: ParameterScheduler, alpha_scheduler: ParameterScheduler,
                 min_priority: float = 1e-3, num_stacked_frames: int = 1,
                 seed: int = None, continuous_actions: bool = False, ):
        self.capacity = capacity

        self.state_shape = state_shape
        self.curr_write_idx = 0
        self.available_samples = 0

        # Memory buffer and priority sum-tree
        self.buffer = ReplayBuffer(state_shape, capacity)
        self.sum_tree = SumTree([0 for _ in range(self.capacity)])

        self.beta_scheduler = beta_scheduler
        self.alpha_scheduler = alpha_scheduler

        self.beta = beta_scheduler.initial
        self.alpha = alpha_scheduler.initial

        self.min_priority = min_priority
        self.num_stacked_frames = num_stacked_frames

        self.continuous_actions = continuous_actions

        if seed:
            set_seed(seed)

    def step_episode(self, episode: int):
        """Update internal memory parameters at the end of an episode

        Args:
            episode (int): The episode number
        """
        self.beta = self.beta_scheduler.get_param(episode)
        self.alpha = self.alpha_scheduler.get_param(episode)
        return True

    def add(self, experience: Experience, priority: float = 0):
        """Add an experience tuple, along with it's priority, to the memory buffer

        Args:
            experience (Experience): A named tuple of experience
            priority (float): The initial priority of the experience tuple

        Add an experience tuple to the memory buffer. The experience tuple is written to the buffer at
        curr_write_idx, and the priority is stored in the sumtree. After the experience is added, the
        current write index is incremented, along with the number of available_samples.
        """
        current_experience = experience.cpu()
        self.buffer[self.curr_write_idx] = current_experience
        self.update(self.curr_write_idx, priority)

        self.curr_write_idx = (self.curr_write_idx + 1) % self.capacity
        # max out available samples at the memory buffer size
        self.available_samples = min(self.available_samples + 1, self.capacity - 1)

    def update(self, indices: Union[int, torch.LongTensor], priorities: Union[float, torch.FloatTensor]):
        """Update the priority value of a node

        Args:
            indices (int): The integer node indices
            priorities (float): The updated priorities
        """
        if isinstance(indices, int):
            indices = np.array([indices])
        if isinstance(priorities, (float, int)):
            priorities = np.array([priorities])

        indices = indices.reshape(-1)
        priorities = priorities.reshape(-1)
        assert len(indices) == len(priorities), "{}, {}".format(len(indices), len(priorities))
        if float(priorities.min()) < 0:
            raise ValueError('Priorities must be > 0')

        priorities = np.power(priorities + self.min_priority, self.alpha)

        for i, idx in enumerate(indices):
            self.sum_tree.update_node(self.sum_tree.leaf_nodes[idx], float(priorities[i]))

    def sample(self, num_samples: int, *args) -> ExperienceBatch:
        """Sample a batch of experience from the memory buffer"""
        sampled_idxs = []
        is_weights = []
        sample_no = 0
        while sample_no < num_samples:
            sample_val = np.random.uniform(0, self.sum_tree.root_node.value)
            sample_node = self.sum_tree.get_node(sample_val, self.sum_tree.root_node)
            sampled_idxs.append(sample_node.idx)
            p = sample_node.value / self.sum_tree.root_node.value
            is_weights.append((self.available_samples + 1) * p)
            sample_no += 1

        # apply the beta factor and normalize so that the maximum is_weight < 1
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, - self.beta)
        # now load up the state and next state variables according to sampled idxs
        states, next_states, actions, rewards, terminal = [], [], [], [], []

        for idx in sampled_idxs:
            experience: Experience = self.buffer[idx]
            states.append(experience.state)
            next_states.append(experience.next_state)

            actions.append(experience.action)
            rewards.append(experience.reward)
            terminal.append(experience.done)

        f = torch.FloatTensor if self.continuous_actions else torch.LongTensor
        experience_batch = ExperienceBatch(
            states=torch.stack(states).float(),
            actions=f(torch.stack(actions)).view(num_samples, -1),
            rewards=torch.FloatTensor(rewards).view(num_samples, 1),
            next_states=torch.stack(next_states).float(),
            dones=torch.LongTensor(terminal).view(num_samples, 1),
            sample_idxs=torch.LongTensor(sampled_idxs).view(num_samples, 1),
            is_weights=torch.from_numpy(is_weights).view(num_samples, 1).float(),
        )
        experience_batch.to(device)
        return experience_batch

    def __len__(self):
        """Return the current size of internal memory."""
        return self.available_samples


class ExtendedPrioritizedMemory(PrioritizedMemory):
    def __init__(self, capacity: int, state_shape: tuple, beta_scheduler: ParameterScheduler, alpha_scheduler: ParameterScheduler,
                 min_priority: float = 1e-7,
                 seed: int = None, continuous_actions: bool = False, ):
        super().__init__(capacity, state_shape, beta_scheduler, alpha_scheduler, min_priority, seed, continuous_actions)

    def sample(self, num_samples: int, *args) -> ExperienceBatch:
        """Sample a batch of experience from the memory buffer

        Firstly, experiences are sampled from the memory sumtree proportionally to their priority
            P(i) = p_i ^ alpha / (Sum_k(p_k^alpha))

        For each sampled experience, the previous num_stacked_frames of state are also obtained, resulting the
        output states to have shape (batch_size, num_stacked_frames, *state_shape)

        Experience samples are sampled from a distribution according to their priority. This is done by
        selecting a uniform random number between 0 and the base node value of the SumTree, then
        this sample value is retrieved from the SumTree data structure according to the stored priorities. Note
        that all priority values in the sumtree are `adjusted' (a small constant is added and the
        value is raised to the power of alpha)

        Args:
            num_samples (int): The number of samples to draw

        Returns:
            states (torch.FloatTensor): Shape (batch_size, num_stacked_frames, *state_shape)
            actions (torch.LongTensor): Shape (batch_size, action_size)
            rewards (torch.FloatTensor): Shape (batch_size, 1)
            next_states (torch.FloatTensor): Shape (batch_size, num_stacked_frames, *state_shape)
            terminal (torch.Tensor): Shape (batch_size, 1)
            sampled_idxs (List[int]): Size of batch_size
            is_weights  (List[float]): Size of batch_size
        """
        sampled_idxs = []
        is_weights = []
        sample_no = 0
        while sample_no < num_samples:
            sample_val = np.random.uniform(0, self.sum_tree.root_node.value)
            sample_node = self.sum_tree.get_node(sample_val, self.sum_tree.root_node)
            # Only include samples with sufficient frames before
            if self.num_stacked_frames - 1 < sample_node.idx < self.available_samples - 1:
                # Account for state and next state; all must be present and in order
                frame_time_steps = [e.t_step for e in self.buffer[sample_node.idx - self.num_stacked_frames + 1: sample_node.idx + 2]]
                if frame_time_steps == sorted(frame_time_steps):
                    sampled_idxs.append(sample_node.idx)
                    p = sample_node.value / self.sum_tree.root_node.value
                    is_weights.append((self.available_samples + 1) * p)
                    sample_no += 1
                else:
                    # This sample is invalid; de-prioritize it
                    self.sum_tree.update_node(sample_node, 0.0)

        # apply the beta factor and normalize so that the maximum is_weight < 1
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, - self.beta)
        # now load up the state and next state variables according to sampled idxs
        states, next_states = [], []

        actions, rewards, terminal = [], [], []

        for idx in sampled_idxs:
            state_frames = torch.cat(
                [e.state for e in self.buffer[idx - self.num_stacked_frames + 1: idx + 1]]
            )
            next_state_frames = torch.cat(
                [e.state for e in self.buffer[idx - self.num_stacked_frames + 2: idx + 2]]
            )

            states.append(state_frames)
            next_states.append(next_state_frames)

            experience_frame: Experience = self.buffer[idx]
            actions.append(experience_frame.action)
            rewards.append(experience_frame.reward)
            terminal.append(experience_frame.done)

        f = torch.FloatTensor if self.continuous_actions else torch.LongTensor
        experience_batch = ExperienceBatch(
            states=torch.stack(states).float(),
            actions=f(torch.stack(actions)).view(num_samples, -1),
            rewards=torch.FloatTensor(rewards).view(num_samples, 1),
            next_states=torch.stack(next_states).float(),
            dones=torch.LongTensor(terminal).view(num_samples, 1),
            sample_idxs=torch.LongTensor(sampled_idxs).view(num_samples, 1),
            is_weights=torch.from_numpy(is_weights).view(num_samples, 1).float(),
        )
        experience_batch.to(device)
        return experience_batch

class MemoryStreams:
    def __init__(self, stream_ids: List[str], capacity, state_shape, beta_scheduler, alpha_scheduler,
                 min_priority: Optional[float] = None, num_stacked_frames=1, seed=None, continuous_actions=False):
        self.streams: Dict[str, PrioritizedMemory] = {}
        if seed:
            set_seed(seed)
        for s in stream_ids:
            self.streams[s] = PrioritizedMemory(
                capacity,
                state_shape,
                beta_scheduler,
                alpha_scheduler,
                min_priority=min_priority,
                num_stacked_frames=num_stacked_frames,
                seed=seed,
                continuous_actions=continuous_actions
            )

    def sample(self, num_samples: int) -> ExperienceBatch:
        # keys, values = [], []
        # for k, memory_stream in self.streams.items():
        #     keys.append(k)
        #     values.append(memory_stream.sum_tree.root_node.value)
        #
        # values = np.array(values)
        # probs = values / np.sum(values)
        #
        # sampled_streams_ = [np.random.choice(keys, p=probs) for _ in range(num_samples)]

        sampled_streams_ = random.choices(list(self.streams.keys()), k=num_samples)
        streams_to_num_samples = Counter(sampled_streams_)
        sampled_streams, states, actions, rewards, next_states, terminal, sampled_idxs, is_weights = [], [], [], [], [], [], [], []

        sampled_streams = []
        for sampled_stream, n_stream_samples in streams_to_num_samples.items():
            experience_batch = self.streams[sampled_stream].sample(n_stream_samples)
            states.extend(experience_batch.states)
            actions.extend(experience_batch.actions)
            rewards.extend(experience_batch.rewards)
            next_states.extend(experience_batch.next_states)
            terminal.extend(experience_batch.dones)
            sampled_idxs.extend(experience_batch.sample_idxs)
            is_weights.extend(experience_batch.is_weights)
            sampled_streams.extend([sampled_stream] * n_stream_samples)

        experience_batch = ExperienceBatch(
            states=torch.stack(states).float(),
            actions=torch.stack(actions).float(),
            rewards=torch.stack(rewards),
            next_states=torch.stack(next_states),
            dones=torch.stack(terminal),
            sample_idxs=torch.stack(sampled_idxs),
            is_weights=torch.stack(is_weights),
            memory_streams=sampled_streams
        )

        experience_batch.shuffle()
        experience_batch.to(device)
        return experience_batch

    def __getitem__(self, stream_name):
        return self.streams[stream_name]

    def __iter__(self):
        for stream_id, memory in self.streams.items():
            yield stream_id, memory

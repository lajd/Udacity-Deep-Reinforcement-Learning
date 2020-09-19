import random
from collections import deque, namedtuple
from typing import List, Dict, Optional
import torch
import numpy as np
from tools.rl_constants import ExperienceBatch, Experience
from tools.misc import set_seed
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.buffer = deque(maxlen=buffer_size)  # internal memory (deque)
        self.capacity = buffer_size
        self.seed = random.seed(seed)

    def update(self, *args):
        pass

    def step_episode(self, i_episode: int):
        pass

    def add(self, experience: Experience):
        """Add a new experience to memory."""
        if experience is not None:
            self.buffer.append(experience.cpu())

    def sample(self, batch_size: int):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=batch_size)
        states = [e.state for e in experiences if e.state is not None]
        actions = [e.action.value for e in experiences if e.action is not None]
        rewards = [e.reward for e in experiences if e.reward is not None]
        next_states = [e.next_state for e in experiences if e.next_state  is not None]
        dones = [e.done for e in experiences if e.done is not None]
        joint_states = [e.joint_state for e in experiences if e.joint_state is not None]
        joint_actions = [e.joint_action for e in experiences if e.joint_action is not None]
        joint_next_states = [e.joint_next_state for e in experiences if e.joint_next_state is not None]

        states = None if len(states) == 0 else torch.from_numpy(np.concatenate(states)).float().to(device)
        actions = None if len(actions) == 0 else torch.from_numpy(np.concatenate(actions)).float().to(device)

        rewards = torch.from_numpy(np.concatenate(rewards)).float().to(device)
        next_states = None if len(next_states) == 0 else torch.from_numpy(np.concatenate(next_states)).float().to(device)
        dones = torch.from_numpy(np.concatenate(dones).astype(np.uint8)).float().to(device)

        joint_states = None if len(joint_next_states) == 0 else torch.from_numpy(np.concatenate(joint_states)).float().to(device)
        joint_actions = None if len(joint_next_states) == 0 else torch.from_numpy(np.concatenate(joint_actions)).float().to(device)
        joint_next_states = None if len(joint_next_states) == 0 else torch.from_numpy(np.concatenate(joint_next_states)).float().to(device)

        return ExperienceBatch(
            states=states, actions=actions, rewards=rewards, next_states=next_states,
            dones=dones, joint_states=joint_states, joint_actions=joint_actions,
            joint_next_states=joint_next_states
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)


class MemoryStreams:
    def __init__(self, stream_ids: List[str], capacity, seed=None):
        self.streams: Dict[str, Memory] = {}
        if seed:
            set_seed(seed)

        for s in stream_ids:
            self.streams[s] = Memory(
                capacity,
                seed
            )

    def sample(self, num_samples: int) -> ExperienceBatch:
        sampled_streams_ = random.choices(list(self.streams.keys()), k=num_samples)
        streams_to_num_samples = Counter(sampled_streams_)
        sampled_streams, states, actions, rewards, next_states, terminal, sampled_idxs, is_weights,\
        joint_states, joint_actions, joint_next_states = [], [], [], [], [], [], [], [], [], [], []

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

            if experience_batch.joint_states:
                joint_states.extend(experience_batch.joint_states)
            if experience_batch.joint_actions:
                joint_actions.extend(experience_batch.joint_actions)
            if experience_batch.joint_next_states:
                joint_next_states.extend(experience_batch.joint_next_states)

        experience_batch = ExperienceBatch(
            states=torch.stack(states).float(),
            actions=torch.stack(actions).float(),
            rewards=torch.stack(rewards),
            next_states=torch.stack(next_states),
            dones=torch.stack(terminal),
            sample_idxs=torch.stack(sampled_idxs),
            is_weights=torch.stack(is_weights),
            memory_streams=sampled_streams,
            joint_states=None if len(joint_states) == 0 else torch.stack(joint_states).float(),
            joint_actions=None if len(joint_actions) == 0 else torch.stack(joint_actions).float(),
            joint_next_states=None if len(joint_next_states) == 0 else torch.stack(joint_next_states).float(),
        )

        experience_batch.shuffle()
        experience_batch.to(device)
        return experience_batch

    def __getitem__(self, stream_name):
        return self.streams[stream_name]

    def __iter__(self):
        for stream_id, memory in self.streams.items():
            yield stream_id, memory

    def __len__(self):
        return min([len(m) for m in self.streams.values()])

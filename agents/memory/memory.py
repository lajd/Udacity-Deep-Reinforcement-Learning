import random
from collections import deque, namedtuple
import torch
import numpy as np
from tools.rl_constants import ExperienceBatch, Experience

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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def update(self, *args):
        pass

    def step_episode(self, i_episode: int):
        pass

    def add(self, experience: Experience):
        """Add a new experience to memory."""
        # e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience.cpu())

    def sample(self, batch_size: int):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        joint_states = torch.from_numpy(np.vstack([e.joint_state for e in experiences if e is not None])).float().to(device)
        joint_actions = torch.from_numpy(np.vstack([e.joint_action for e in experiences if e is not None])).float().to(device)
        joint_next_states = torch.from_numpy(np.vstack([e.joint_next_state for e in experiences if e is not None])).float().to(device)

        return ExperienceBatch(
            states=states, actions=actions, rewards=rewards, next_states=next_states,
            dones=dones, joint_states=joint_states, joint_actions=joint_actions,
            joint_next_states=joint_next_states
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

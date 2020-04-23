import numpy as np
import torch
from tools.data_structures.sumtree import SumTree
from tools.experience import Experience

MAX_EPSILON = 1
MIN_EPSILON = 0.1
EPSILON_MIN_ITER = 500000
GAMMA = 0.99
BATCH_SIZE = 32
TAU = 0.08
POST_PROCESS_IMAGE_SIZE = (105, 80, 1)
DELAY_TRAINING = 50000
BETA_DECAY_ITERS = 500000
MIN_BETA = 0.4
MAX_BETA = 1.0
NUM_FRAMES = 4
GIF_RECORDING_FREQ = 100
MODEL_SAVE_FREQ = 100


class Memory(object):
    def __init__(self, capacity: int, state_shape: tuple, beta: float = 0.4, alpha: float = 0.6,
                 min_priority: float = 0.01):
        self.capacity = capacity

        self.state_shape = state_shape
        self.curr_write_idx = 0
        self.available_samples = 0

        self.buffer = [(np.zeros(state_shape, dtype=np.float32), 0.0, 0.0, 0.0) for _ in range(self.capacity)]

        self.sum_tree = SumTree([0 for i in range(self.capacity)])

        self.frame_idx = 0
        self.action_idx = 1
        self.reward_idx = 2
        self.terminal_idx = 3

        self.beta = beta
        self.alpha = alpha
        self.min_priority = min_priority

    def add(self, experience: tuple, priority: float):
        """
        Add an experience tuple to the memory buffer
        The experience tuple is written to the buffer at curr_write_idx and the priority is sent to the update method of the class.

        After the experience tuple is added, the current write index is incremented. If the current write index now exceeds
        the size of the buffer, it is reset back to 0 to start overwriting old experience tuples. Next, the available_samples
        value is incremented, but only if it is less than the size of the memory, otherwise it is clipped at the size of the
        memory.
        :param experience:
        :param priority:
        :return:
        """
        self.buffer[self.curr_write_idx] = experience
        self.update(self.curr_write_idx, priority)
        self.curr_write_idx += 1
        # reset the current writer position index if creater than the allowed size
        if self.curr_write_idx >= self.capacity:
            self.curr_write_idx = 0
        # max out available samples at the memory buffer size
        if self.available_samples + 1 < self.capacity:
            self.available_samples += 1
        else:
            self.available_samples = self.capacity - 1

    def update(self, idx: int, priority: float):
        """
        update the priority value in the SumTree:

        he update method of the Memory class in turn calls the SumTree update function
        which is outside this class. Notice that the “raw” priority is not passed to the SumTree
        update, but rather the “raw” priority is first passed to the adjust_priority method.
        This method adds the minimum priority factor and then raises the priority to the power of
        i.e. it performs the following calculations:

        p_i = |delta_i| + eps
        P(i) = p_i^alpha / Sum_k(p_k^alpha)
        :param idx:
        :param priority:
        :return:
        """
        self.sum_tree.update_node(self.sum_tree.leaf_nodes[idx], self.adjust_priority(priority))

    def adjust_priority(self, priority: float):
        return np.power(priority + self.min_priority, self.alpha)

    def sample(self, num_samples: int):
        """
        The purpose of this method is to perform priority sampling of the experience buffer, but also to calculate the
        importance sampling weights for use in the training steps. The first step is a while loop which iterates until
        num_samples have been sampled. This sampling is performed by selecting a uniform random number between 0 and the
        base node value of the SumTree. This sample value is then retrieved from the SumTree data structure according to
        the stored priorities.

        A check is then made to ensure that the sampled index is valid and if so it is appended to a list of sampled indices.
        After this appending, the value is calculated. The SumTree base node value is actually the sum of all priorities
        of samples stored to date. Also recall that the  value has already been applied to all samples as the “raw”
        priorities are added to the SumTree.
        :param num_samples:
        :return:
        """
        sampled_idxs = []
        is_weights = []
        sample_no = 0
        while sample_no < num_samples:
            sample_val = np.random.uniform(0, self.sum_tree.root_node.value)
            samp_node = self.sum_tree.get_node(sample_val, self.sum_tree.root_node)
            if NUM_FRAMES - 1 < samp_node.idx < self.available_samples - 1:
                sampled_idxs.append(samp_node.idx)
                p = samp_node.value / self.sum_tree.root_node.value
                is_weights.append((self.available_samples + 1) * p)
                sample_no += 1
        # apply the beta factor and normalise so that the maximum is_weight < 1
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, -self.beta)
        is_weights = is_weights / np.max(is_weights)
        # now load up the state and next state variables according to sampled idxs
        states = np.zeros((num_samples, *self.state_shape, NUM_FRAMES),
                          dtype=np.float32)
        next_states = np.zeros((num_samples, *self.state_shape, NUM_FRAMES),
                               dtype=np.float32)

        actions, rewards, terminal = [], [], []
        for i, idx in enumerate(sampled_idxs):
            for j in range(NUM_FRAMES):
                states[i, :, :, j] = self.buffer[idx + j - NUM_FRAMES + 1][self.frame_idx][:, :, 0]
                next_states[i, :, :, j] = self.buffer[idx + j - NUM_FRAMES + 2][self.frame_idx][:, :, 0]
            actions.append(self.buffer[idx][self.action_idx])
            rewards.append(self.buffer[idx][self.reward_idx])
            terminal.append(self.buffer[idx][self.terminal_idx])
        return states, np.array(actions), np.array(rewards), next_states, np.array(terminal), sampled_idxs, is_weights

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)


# from collections import namedtuple, deque
# import random
#
# import torch
# import numpy as np
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""
#
#     def __init__(self, action_size, buffer_size, batch_size):
#         """Initialize a ReplayBuffer object.
#
#         Params
#         ======
#             action_size (int): dimension of each action
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#         """
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)
#
#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         experiences = random.sample(self.memory, k=self.batch_size)
#
#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
#             device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
#             device)
#
#         return (states, actions, rewards, next_states, dones)
#
#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)

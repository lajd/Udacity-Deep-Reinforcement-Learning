import os
import numpy as np
from typing import Tuple, Optional
from agents.base import Agent
from agents.policies.base_policy import Policy
from copy import deepcopy
import torch
from agents.memory.prioritized_memory import PrioritizedMemory
from tools.rl_constants import Experience
from torch.optim.lr_scheduler import _LRScheduler
from tools.misc import set_seed
from agents.models.base import BaseModel
from tools.misc import soft_update
from tools.rl_constants import ExperienceBatch, Action


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_shape: Tuple[int, ...],
                 action_size: int,
                 model: BaseModel,
                 policy: Policy,
                 memory: PrioritizedMemory,
                 lr_scheduler: _LRScheduler,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int = 32,
                 gamma: float = 0.95,
                 tau: float = 1e-3,
                 update_frequency: int = 5,
                 seed: int = None,
                 action_repeats: int = 1,
                 gradient_clip: float = 1,
                 ):
        """Initialize an Agent object.

        Args:
            state_shape (Tuple[int, ...]): Shape of the state
            action_size (int): Number of possible integer actions
            model (torch.nn.Module): Model producing actions from state
            policy (Policy):
            memory: Memory,
            lr_scheduler: _LRScheduler,
            optimizer: torch.optim.Optimizer,
            batch_size: int = 32,
            gamma: float = 0.95,
            tau: float = 1e-3,
            update_frequency: int = 5,
            seed: int = None
        """
        super().__init__(action_size=action_size, state_shape=state_shape)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_frequency = update_frequency
        self.gradient_clip = gradient_clip

        self.previous_action: Optional[Action] = None
        self.action_repeats = action_repeats

        # Double DQN
        self.online_qnetwork = model.to(device)
        self.target_qnetwork = deepcopy(model).to(device).eval()

        self.memory = memory

        self.losses = []

        self.policy: Policy = policy
        self.optimizer: optimizer = optimizer
        self.lr_scheduler: _LRScheduler = lr_scheduler

        if seed:
            set_seed(seed)
            self.online_qnetwork.set_seed(seed)
            self.target_qnetwork.set_seed(seed)

    def set_mode(self, mode: str):
        if mode == 'train':
            self.online_qnetwork.train()
            self.target_qnetwork.train()
            self.policy.train()
        elif mode == 'eval':
            self.online_qnetwork.eval()
            self.target_qnetwork.eval()
            self.policy.eval()
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

    def load(self, path_to_online_network_pth: str):
        assert os.path.exists(path_to_online_network_pth), "Path does not exist"
        self.online_qnetwork.load_state_dict(torch.load(path_to_online_network_pth))

    def preprocess_state(self, state: torch.Tensor):
        preprocessed_state = self.online_qnetwork.preprocess_state(state)
        return preprocessed_state

    def step_episode(self, episode: int, param_frequency: int = 10,  *args):
        self.episode_counter += 1
        self.policy.step_episode(episode)
        self.lr_scheduler.step()
        self.memory.step_episode(episode)
        self.online_qnetwork.step_episode(episode)
        self.target_qnetwork.step_episode(episode)
        return True

    def step(self, experience: Experience, **kwargs) -> None:
        """Step the agent in response to a change in environment"""
        # Add the experience, defaulting the priority 0
        self.memory.add(experience)

        if self.warmup:
            return
        else:
            self.t_step += 1
            # If enough samples are available in memory, get random subset and learn
            if self.t_step % self.update_frequency == 0 and len(self.memory) > self.batch_size:
                experience_batch = self.memory.sample(self.batch_size)
                experience_batch = experience_batch.to(device)

                loss, errors = self.learn(experience_batch)

                with torch.no_grad():
                    if errors.min() < 0:
                        raise RuntimeError("Errors must be > 0, found {}".format(errors.min()))

                    priorities = errors.detach().cpu().numpy()
                    self.memory.update(experience_batch.sample_idxs, priorities)

                # Perform any post-backprop updates
                self.online_qnetwork.step()
                self.target_qnetwork.step()
                self.param_capture.add('loss', loss)

    def get_action(self, state: torch.Tensor, *args, **kwargs) -> Action:
        """Returns actions for given state as per current policy.

        Args:
            state (np.array): Current environment state

        Returns:
            action (int): The action to perform
        """
        state = state.to(device)
        state.requires_grad = False

        if not self.training:
            # Run in evaluation mode
            action: Action = self.policy.get_action(state=state, model=self.online_qnetwork)
        else:
            if not self.previous_action or self.t_step % self.action_repeats == 0:
                # Get the action from the policy
                action: Action = self.policy.get_action(state=state, model=self.online_qnetwork)
                self.previous_action = action
            else:
                # Repeat the last action
                action: Action = self.previous_action

        return action

    def get_random_action(self, state: torch.Tensor, *args, **kwargs) -> Action:
        action = np.array(np.random.random_integers(0, self.action_size - 1, (1, )))
        action = Action(value=action)
        return action

    def learn(self, experience_batch: ExperienceBatch) -> tuple:
        """Update value parameters using given batch of experience tuples and return TD error

        Args:
            experience_batch (ExperienceBatch): Minibatch of experience

        Returns:
            td_errors (torch.FloatTensor): The TD errors for each sample
        """

        # By default, calculate TD errors. Some DQN modifications (eg. categorical DQN) use custom errors/loss
        loss, errors = self.policy.compute_errors(
            self.online_qnetwork,
            self.target_qnetwork,
            experience_batch,
            gamma=self.gamma
        )
        assert errors.min() >= 0

        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online_qnetwork.parameters():
            param.grad.data.clamp_(-self.gradient_clip, self.gradient_clip)
        self.optimizer.step()

        # Perform a soft update of the target -> local network
        soft_update(self.online_qnetwork, self.target_qnetwork, self.tau)
        return loss, errors

import os
import numpy as np
from typing import Tuple
from agents.base import Agent
from agents.policies.base import Policy
from copy import deepcopy
import torch
from agents.memory.prioritized_memory import Memory
from tools.rl_constants import Experience
from torch.optim.lr_scheduler import _LRScheduler
from tools.misc import set_seed
from agents.models.base import BaseModel
from tools.rl_constants import Action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_FRAMES = 4


class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_shape: Tuple[int, ...],
                 action_size: int,
                 model: BaseModel,
                 policy: Policy,
                 memory: Memory,
                 lr_scheduler: _LRScheduler,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int = 32,
                 gamma: float = 0.95,
                 tau: float = 1e-3,
                 update_frequency: int = 5,
                 warmup_steps: int = 0,
                 seed: int = None,
                 action_repeats: int = 1,
                 gradient_clip: float = 1
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
            warmup_steps: int = 0,
            seed: int = None
        """
        super().__init__(
            state_shape=state_shape,
            action_size=action_size,
            policy=policy,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer
        )
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_frequency = update_frequency
        self.warmup_steps = warmup_steps
        self.gradient_clip = gradient_clip

        self.previous_action = None
        self.action_repeats = action_repeats

        # Double DQN
        self.online_qnetwork = model.to(device)
        self.target_qnetwork = deepcopy(model).to(device).eval()

        self.memory = memory

        self.t_step = 0
        self.episode_step = 0
        self.losses = []

        if seed:
            set_seed(seed)
            self.online_qnetwork.set_seed(seed)
            self.target_qnetwork.set_seed(seed)

    def save(self, save_pth: str):
        torch.save(self.online_qnetwork.state_dict(), save_pth)

    def load_pretrained(self, path_to_online_network_pth: str):
        assert os.path.exists(path_to_online_network_pth), "Path does not exist"
        self.online_qnetwork.load_state_dict(torch.load(path_to_online_network_pth))

    def preprocess_state(self, state: torch.Tensor):
        preprocessed_state = self.online_qnetwork.preprocess_state(state)
        return preprocessed_state

    def step_episode(self, episode: int, param_frequency: int = 10):
        self.episode_step += 1
        self.policy.step(episode)
        self.lr_scheduler.step()
        self.memory.step(episode)

        self.online_qnetwork.step_episode(episode)
        self.target_qnetwork.step_episode(episode)
        return True

    def step(self, experience: Experience, **kwargs) -> None:
        """Step the agent in response to a change in environment"""
        self.t_step += 1

        # Add the experience, defaulting the priority 0
        # self.memory.add(experience, 0)
        self.memory.add(experience)

        # If enough samples are available in memory, get random subset and learn
        if self.t_step > self.warmup_steps and self.t_step % self.update_frequency == 0 and len(self.memory) > self.batch_size:
            beta = 0.6 + self.episode_step/1200
            state_frames, actions, rewards, next_state_frames, terminal, idxs, is_weights = self.memory.sample(self.batch_size, beta)
            experiences = (state_frames, actions, rewards, next_state_frames, terminal)
            loss, errors = self.learn(experiences, self.gamma, sample_weights=is_weights)

            with torch.no_grad():
                if errors.min() < 0:
                    raise RuntimeError("Errors must be > 0, found {}".format(errors.min()))

                priorities = errors.detach().cpu().numpy()
                self.memory.update(idxs, priorities)

            # Perform any post-backprop updates
            self.online_qnetwork.step()
            self.target_qnetwork.step()

            self.param_capture.add('loss', loss)

    def act(self, state: torch.Tensor) -> Action:
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
            action = self.policy.get_action(state=state, model=self.online_qnetwork)
        else:
            if self.t_step < self.warmup_steps:
                # Take a random action
                action = Action(value=np.random.randint(0, self.action_size), distribution=None)
            elif not self.previous_action or self.t_step % self.action_repeats == 0:
                # Get the action from the policy
                action = self.policy.get_action(state=state, model=self.online_qnetwork)
                self.previous_action = action
            else:
                # Repeat the last action
                action = self.previous_action
        return action

    def learn(self, experiences: Tuple[np.array, ...], gamma: float, sample_weights: np.array) -> tuple:
        """Update value parameters using given batch of experience tuples and return TD error

        Args:
            experiences (Tuple[np.array, ...]): Tuple of (states, actions, rewwards, next_states, terminal)
            gamma (float): The discount factor to apply to future experiences
            sample_weights (np.array): The weights to apply to each experience

        Returns:
            td_errors (torch.FloatTensor): The TD errors for each sample
        """

        # By default, calculate TD errors. Some DQN modifications (eg. categorical DQN) use custom errors/loss
        assert sample_weights.min() >= 0, "Sample weights must be positive, {}".format(sample_weights.min())
        loss, errors = self.policy.compute_errors(
            self.online_qnetwork,
            self.target_qnetwork,
            experiences,
            error_weights=sample_weights,
            gamma=gamma
        )
        assert errors.min() >= 0

        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online_qnetwork.parameters():
            param.grad.data.clamp_(-self.gradient_clip, self.gradient_clip)
        self.optimizer.step()

        # Perform a soft update of the target -> local network
        self.soft_update(self.online_qnetwork, self.target_qnetwork, self.tau)
        return loss, errors

    @staticmethod
    def soft_update(online_model, target_model, tau) -> None:
        """Soft update model parameters from local to target network.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            online_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

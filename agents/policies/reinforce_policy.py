import torch
import numpy as np
from typing import Optional
from tools.misc import set_seed
from tools.rl_constants import Trajectories
from tools.parameter_decay import ParameterScheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RIGHT = 4
LEFT = 5


class ReinforcePolicy:
    def __init__(
            self,
            beta_scheduler: ParameterScheduler,
            gamma: float = 0.99,
            seed: Optional[int] = None,
    ):

        self.beta_scheduler = beta_scheduler
        if seed:
            self.set_seed(seed)

        # Initialize epsilon
        self.beta = self.beta_scheduler.initial
        self.gamma = gamma

    def step_episode(self, episode):
        self.beta = self.beta_scheduler.get_param(episode)

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    # convert states to probability, passing through the policy
    def _states_to_prob(self, model, states):
        states = torch.stack(states)
        policy_input = states.view(-1, *states.shape[-3:])
        return model(policy_input).view(states.shape[:-3])

    # clipped surrogate function
    # similar as -policy_loss for REINFORCE, but for PPO
    def compute_errors(self, model, trajectories: Trajectories):
        discount = self.gamma ** np.arange(len(trajectories.rewards))
        rewards = np.asarray(trajectories.rewards) * discount[:, np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(trajectories.actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(trajectories.policy_outputs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = self._states_to_prob(model, trajectories.states)
        new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)

        # ratio for clipping
        ratio = new_probs / old_probs

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

        errors = ratio * rewards + self.beta * entropy
        loss = torch.mean(errors)

        loss = - loss  # Gradient ascent
        return loss, errors

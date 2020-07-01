import torch
from torch.optim.lr_scheduler import _LRScheduler
from tools.misc import set_seed
from agents.models.base import BaseModel
from tools.rl_constants import Trajectories

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReinforceAgent:
    """Interacts with and learns from the environment."""
    def __init__(self,
                 model: BaseModel,
                 policy,
                 lr_scheduler: _LRScheduler,
                 optimizer: torch.optim.Optimizer,
                 seed: int = None,
                 gradient_clip: float = 1,
                 ):
        """Initialize an Agent object.

        Args:
            model (torch.nn.Module): Model producing actions from state
            policy:
            optimizer: torch.optim.Optimizer,
            seed: int = None
        """
        self.optimizer = optimizer
        self.policy = policy
        self.gradient_clip = gradient_clip
        self.lr_scheduler = lr_scheduler
        self.model = model.to(device)

        self.t_step = 0
        self.episode_step = 0
        self.losses = []

        if seed:
            set_seed(seed)
            self.model.set_seed(seed)

    def step_episode(self, episode: int):
        self.episode_step += 1
        self.policy.step_episode(episode)
        self.lr_scheduler.step()
        self.model.step_episode(episode)
        return True

    def step(self, trajectories: Trajectories, **kwargs) -> None:
        """Step the agent in response to a change in environment"""
        self.learn(trajectories)
        self.model.step()

    def learn(self, trajectories: Trajectories) -> None:
        """Update value parameters using given batch of experience tuples and return TD error

        Args:
            Trajectories Trajectories

        Returns:
            td_errors (torch.FloatTensor): The TD errors for each sample
        """

        loss, errors = self.policy.compute_errors(self.model, trajectories)
        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-self.gradient_clip, self.gradient_clip)
        self.optimizer.step()
        del loss, errors

    def save(self, save_path: str):
        # save your policy!
        torch.save(self.model, save_path)

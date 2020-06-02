import numpy as np
import torch
from agents.policies.base import Policy
import torch.nn.functional as F
from torch.autograd import Variable
from tools.rl_constants import Action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CategoricalDQNPolicy(Policy):
    """ The Categorical DQN policy creates a distribution over Q(s,a)

    Code is adapted from  https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb
    """
    def __init__(self, action_size: int, num_atoms: int = 51, v_min: float = -10, v_max: float = 10, seed: int = None):
        super().__init__(action_size=action_size)
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(device)
        self.delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

        if seed:
            self.set_seed(seed)

    def get_action(self, state: np.array, model: torch.nn.Module) -> Action:
        """ Implement this function for speed"""
        model.eval()
        with torch.no_grad():
            selected_action = model(state, act=True).argmax()
            selected_action = int(selected_action.detach().cpu().numpy())
            action = Action(value=selected_action, distribution=None)
        model.train()
        return action

    def projection_distribution(self, target_model: torch.nn.Module, next_state: torch.Tensor, rewards: torch.Tensor,
                                dones: torch.Tensor, gamma: float):
        with torch.no_grad():
            batch_size = next_state.size(0)
            next_action = target_model(next_state).argmax(1)
            next_dist = target_model.dist(next_state)
            next_dist = next_dist[range(batch_size), next_action]

            Tz = rewards + (1 - dones) * gamma * self.support
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long() \
                .unsqueeze(1).expand(batch_size, self.num_atoms).to(device)

            proj_dist = torch.zeros(next_dist.size()).to(device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
            return proj_dist

    def compute_errors(self, online_model, target_model, experiences: tuple, error_weights: torch.FloatTensor, gamma: float = 0.99) -> tuple:
        states, actions, rewards, next_states, dones = experiences
        batch_size = states.shape[0]
        dist = online_model.dist(states)
        log_p = torch.log(dist[range(batch_size), actions.view(-1)])
        target_dist = self.projection_distribution(target_model, next_states, rewards, dones, gamma)
        errors = - (target_dist * log_p).sum(1)
        assert 0 <= errors.min(), errors.min()
        loss = (errors * error_weights).mean()
        return loss, errors

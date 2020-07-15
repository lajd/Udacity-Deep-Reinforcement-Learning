import numpy as np
import torch
from agents.policies.base_policy import Policy


class MaxPolicy(Policy):
    """ The Max policy used with the Noisy DQN flavour

    The selected action is simply the action with the largest Q(s,a) value
    """
    def __init__(self, action_size: int, seed: int = None):
        super().__init__(action_size=action_size)

        if seed:
            self.set_seed(seed)

    def get_action(self, state: np.array, model: torch.nn.Module) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            action_values = model.forward(state, act=True)
        model.train()

        action = action_values.max(1)[1].cpu().numpy()
        return action

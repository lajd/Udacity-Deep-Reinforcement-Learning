import numpy as np
import torch
from agents.policies.base_policy import Policy
from tools.rl_constants import Action


class SoftmaxPolicy(Policy):
    def __init__(self, action_size: int, seed: int = None):
        super().__init__(action_size=action_size)
        self.action_size = action_size

        if seed:
            self.set_seed(seed)

    def get_action(self, state: np.array, model: torch.nn.Module) -> Action:
        """ Implement this function for speed"""

        model.eval()
        with torch.no_grad():
            action_values = model.forward(state, act=True)
        model.train()

        probs = torch.nn.functional.softmax(action_values)
        action = Action(value=int(np.random.choice(np.arange(0, self.action_size), p=probs.view(-1).numpy())), distribution=probs.data.cpu().numpy())
        return action

    def get_deterministic_policy(self, state_action_values_dict: dict):
        deterministic_policy = {}
        for state in state_action_values_dict:
            deterministic_policy[state] = np.argmax(state_action_values_dict[state])
        return deterministic_policy

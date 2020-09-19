import torch
from typing import Optional, Tuple, Callable
from tools.rl_constants import ExperienceBatch, Action
from agents.policies.ddpg_policy import DDPGPolicy
from agents.models.components.noise import Noise, GaussianNoise
from tools.parameter_scheduler import ParameterScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3Policy(DDPGPolicy):
    """ Policy for the TD3 algorithm"""
    def __init__(
            self,
            random_brain_action_factory: Callable,
            action_dim: int,
            gamma: float = 0.99,
            seed: Optional[int] = None,
            action_range: Tuple[int, int] = (-1, 1),
            noise: Optional[Noise] = None,
            epsilon_scheduler: Optional[ParameterScheduler] = None,
    ):
        if not (noise or epsilon_scheduler):
            raise ValueError("Must provide either noise or epsilon_scheduler")

        super().__init__(action_dim=action_dim, noise=noise, gamma=gamma, seed=seed,
                         action_range=action_range, random_brain_action_factory=random_brain_action_factory, epsilon_scheduler=epsilon_scheduler)
        self.gaussian_noise = GaussianNoise()

    def compute_actor_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic) -> tuple:
        """ Compute the error and loss of the actor"""
        # Compute actor loss
        actor_errors = - online_critic.qa(experience_batch.states, online_actor(experience_batch.states))
        actor_loss = actor_errors.mean()
        return actor_loss, actor_errors

    def compute_critic_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic) -> tuple:
        """ Compute the error and loss of the critic"""

        with torch.no_grad():
            next_actions = target_actor(experience_batch.next_states)
            # Smooth the targets used for policy updates
            # Add noise to the actions used to calculate the target & clip
            next_actions += self.gaussian_noise.sample(next_actions).float().to(device)
            next_actions = next_actions.clamp(self.action_range[0], self.action_range[1])

            # Compute the target Q value
            target_q1, target_q2 = target_critic(experience_batch.next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2)
            target_q = experience_batch.rewards + (1 - experience_batch.dones) * self.gamma * min_target_q

        current_q1, current_q2 = online_critic(experience_batch.states, experience_batch.actions)

        td_errors_a = torch.abs(current_q1 - target_q)
        td_errors_b = torch.abs(current_q2 - target_q)

        if experience_batch.is_weights is not None:
            norm_is_weights = experience_batch.get_norm_is_weights()
            td_errors_a *= norm_is_weights.view_as(td_errors_a)
            td_errors_b *= norm_is_weights.view_as(td_errors_b)

        td_errors = td_errors_a + td_errors_b
        # Compute critic loss (joint loss between both critic streams)
        critic_loss = torch.pow(td_errors_a, 2).mean() + torch.pow(td_errors_b, 2).mean()

        return critic_loss, td_errors

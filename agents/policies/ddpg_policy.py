import torch
import numpy as np
from typing import Optional, Tuple, Callable
from tools.misc import set_seed
from tools.rl_constants import ExperienceBatch, Action
from tools.parameter_scheduler import ParameterScheduler
import torch.nn.functional as F
from agents.policies.base_policy import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGPolicy(Policy):
    """ Policy for the DDPG algorithm"""
    def __init__(
            self,
            noise,
            random_brain_action_factory: Callable,
            action_dim: int,
            gamma: float = 0.99,
            seed: Optional[int] = None,
            action_range: Tuple[float, float] = (-1, 1),
            epsilon_scheduler: Optional[ParameterScheduler] = None,
    ):
        super().__init__(action_dim, seed=seed)
        self.gamma = gamma
        self.noise = noise
        self.action_range = action_range
        self.action_dim = action_dim

        self.random_brain_action_generator = random_brain_action_factory()
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = epsilon_scheduler.initial if epsilon_scheduler else None

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    def step_episode(self, episode: int):
        """ Perform any end-of-episode updates """
        if self.epsilon_scheduler:
            self.epsilon = self.epsilon_scheduler.get_param(episode)
        if self.noise:
            self.noise.reset()

    def get_action(self, state: torch.Tensor, online_actor: torch.nn.Module) -> Action:
        """Returns actions for given state as per current policy."""
        def get_actions_():
            online_actor.eval()
            with torch.no_grad():
                actions_ = online_actor(state)
            online_actor.train()
            return actions_

        if self.epsilon_scheduler:
            if self.training:
                r = np.random.random()
                if r <= self.epsilon:
                    action = self.random_brain_action_generator.sample()
                else:
                    action = get_actions_().cpu().data.numpy()
                    if self.random_brain_action_generator.continuous_actions:
                        action = np.clip(
                            action,
                            self.random_brain_action_generator.continuous_action_range[0],
                            self.random_brain_action_generator.continuous_action_range[1],
                        )  # epsilon greedy policy
            else:
                action = get_actions_().cpu().data.numpy()
        elif self.noise:
            action = get_actions_().cpu().data.numpy()
            if self.training:
                action += self.noise.sample(action)
            action = np.clip(action, self.action_range[0], self.action_range[1])
        else:
            raise ValueError('Must provide either epsilon_scheduler or noise')

        return Action(value=action)

    def get_random_action(self, *args) -> Action:
        """ Get a random action (used for warmup) """
        action = self.random_brain_action_generator.sample()
        action = Action(value=action)
        return action

    def compute_actor_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic) -> tuple:
        """ Compute the error and loss of the actor"""
        actions_pred = online_actor(experience_batch.states)
        actor_errors = - online_critic(experience_batch.states, actions_pred)
        actor_loss = actor_errors.mean()
        return actor_loss, actor_errors

    def compute_critic_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic) -> tuple:
        """ Compute the error and loss of the critic"""
        # Calculate the critic errors/loss
        # Get predicted next-state actions and Q values from target models
        actions_next = target_actor(experience_batch.next_states)
        next_q_targets = target_critic(experience_batch.next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = experience_batch.rewards + (self.gamma * next_q_targets * (1 - experience_batch.dones))
        # Compute critic loss
        q_expected = online_critic(experience_batch.states, experience_batch.actions)
        td_errors = torch.abs(q_expected - q_targets)
        if experience_batch.is_weights is not None:
            norm_is_weights = experience_batch.get_norm_is_weights()
            td_errors *= norm_is_weights.view_as(td_errors)
        critic_loss = torch.pow(td_errors, 2).mean()
        return critic_loss, td_errors

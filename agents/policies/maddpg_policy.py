import torch
import numpy as np
from typing import Optional, List, Callable
from tools.misc import set_seed
from tools.rl_constants import ExperienceBatch, Action
from tools.parameter_scheduler import ParameterScheduler
from agents.policies.base_policy import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGPolicy(Policy):
    """ Policy for the MADDPG algorithm"""
    def __init__(
            self,
            noise_factory,
            num_agents: int,
            critic_input_dim: int,
            action_dim: int,
            map_agent_to_state_slice: list,
            map_agent_to_action_slice: list,
            epsilon_scheduler: ParameterScheduler,
            random_brain_action_factory: Callable,
            gamma: float = 0.99,
            seed: Optional[int] = None,
    ):
        super().__init__(action_dim, seed=seed)
        self.gamma = gamma
        self.noise = noise_factory()
        self.num_agents = num_agents
        self.action_dim = action_dim

        self.epsilon_scheduler = epsilon_scheduler

        self.critic_input_shape = critic_input_dim

        self.t_step = 0
        self.epsilon = self.epsilon_scheduler.initial
        self.random_action_generator = random_brain_action_factory()

        self.slice_cache = {}

        self.map_agent_to_state_slice = map_agent_to_state_slice

        self.map_agent_to_action_slice = map_agent_to_action_slice

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    def step(self, episode_number: int):
        pass

    def step_episode(self, episode: int):
        """ Perform any end-of-episode updates """
        self.epsilon = self.epsilon_scheduler.get_param(episode)
        self.noise.reset()

    def get_action(self, state: torch.Tensor, online_actor: torch.nn.Module, training: bool = False) -> Action:
        """Returns actions for given state as per current policy."""

        def get_actions_():
            online_actor.eval()
            with torch.no_grad():
                actions_ = online_actor(state)
            online_actor.train()
            return actions_

        if training:
            r = np.random.random()
            if r <= self.epsilon:
                action = self.random_action_generator.sample()
            else:
                action = get_actions_().cpu().data.numpy()
                if self.random_action_generator.continuous_actions:
                    action = np.clip(
                        action,
                        self.random_action_generator.continuous_action_range[0],
                        self.random_action_generator.continuous_action_range[1],
                    )  # epsilon greedy policy
        else:
            action = get_actions_().cpu().data.numpy()

        action = Action(value=action)
        return action

    def get_random_action(self, *args) -> Action:
        """ Get a random action (used for warmup) """
        action = self.random_action_generator.sample()
        action = Action(value=action)
        return action

    def get_other_agent_atributes(self, x: torch.Tensor, agent_number: int, slicing_list: List[Callable]):
        output = []
        for k, f in enumerate(slicing_list):
            if k != agent_number:
                output.append(f(x))
        output = torch.cat(output, dim=1)
        return output

    def get_agent_attributes(self, x: torch.Tensor, agent_number: int, slicing_list: List[Callable]):
        return slicing_list[agent_number](x)

    def compute_actor_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic, agent_num, *args, **kwargs) -> tuple:
        """ Compute the error and loss of the actor"""

        other_agent_states = self.get_other_agent_atributes(
            experience_batch.joint_states,
            agent_num,
            self.map_agent_to_state_slice
        )

        other_agent_actions = online_actor(other_agent_states).detach().float()

        # Impose structure in the critic input to help with learning, of the in the form
        # [<agent_state> <other_agent_states>, <other_agent_actions>]
        agent_action = online_actor(experience_batch.states)

        actor_errors = - online_critic(
            experience_batch.states,
            other_agent_states,
            other_agent_actions,
            agent_action,
        )
        actor_loss = actor_errors.mean()
        return actor_loss, actor_errors

    def compute_critic_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic, agent_num, *args, **kwargs) -> tuple:
        """ Compute the error and loss of the critic"""

        other_agent_next_states = self.get_other_agent_atributes(
            experience_batch.joint_next_states,
            agent_num,
            self.map_agent_to_state_slice
        )

        all_other_agent_next_actions = target_actor(other_agent_next_states)
        all_agent_next_actions = target_actor(experience_batch.next_states)

        with torch.no_grad():
            q_target_next = target_critic(
                experience_batch.next_states, other_agent_next_states,
                all_other_agent_next_actions.float(), all_agent_next_actions
            )

        q_targets = experience_batch.rewards + (self.gamma * q_target_next * (1 - experience_batch.dones))

        other_agent_actions = self.get_other_agent_atributes(
            experience_batch.joint_actions,
            agent_num,
            self.map_agent_to_action_slice
        )

        other_agent_states = self.get_other_agent_atributes(
            experience_batch.joint_states,
            agent_num,
            self.map_agent_to_state_slice
        )

        q_expected = online_critic(
            experience_batch.states,
            other_agent_states,
            other_agent_actions,
            experience_batch.actions.float()
        )

        # critic loss
        huber_errors = torch.nn.SmoothL1Loss(reduction='none')
        td_errors = huber_errors(q_expected, q_targets.detach())
        critic_loss = td_errors.mean()

        return critic_loss, td_errors

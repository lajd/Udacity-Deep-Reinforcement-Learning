import torch
import numpy as np
from typing import Optional, Callable
from tools.misc import set_seed
from tools.rl_constants import ExperienceBatch, RandomBrainAction, Action
from tools.parameter_scheduler import ParameterScheduler
from agents.models.components.noise import GaussianNoise
from agents.policies.base_policy import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IndependentMADDPGPolicy(Policy):
    """ Policy for the (independend) MADDPG algorithm"""
    def __init__(
            self,
            agent_id: str,
            brain_set,
            action_dim: int,
            map_agent_to_state_slice: dict,
            map_agent_to_action_slice: dict,
            random_brain_action_factory: lambda: RandomBrainAction,
            epsilon_scheduler: ParameterScheduler=ParameterScheduler(initial=1, lambda_fn=lambda i: 0.95**i, final=0.01),
            gamma: float = 0.99,
            seed: Optional[int] = None,
            continuous_actions_clip_range: Optional[tuple] = (-1, 1),
            matd3: bool = False,
            gaussian_noise_factory: Callable = lambda: GaussianNoise(),
            continuous_actions: bool = True
    ):
        super().__init__(action_dim, seed=seed)
        if seed:
            self.set_seed(seed)

        self.agent_id = agent_id
        self.gamma = gamma

        self.brain_set = brain_set

        self.action_dim = action_dim

        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon = epsilon_scheduler.initial

        self.random_action_generator = random_brain_action_factory()
        self.gaussian_noise = gaussian_noise_factory()

        self.map_agent_to_state_slice = map_agent_to_state_slice

        self.map_agent_to_action_slice = map_agent_to_action_slice

        self.continuous_actions_clip_range = continuous_actions_clip_range
        self.continuous_actions = continuous_actions

        self.matd3 = matd3

        self.target_actor_map = {}
        for brain_name, brain in self.brain_set:
            for i, agent in enumerate(brain.agents):
                key = "{}_{}".format(brain_name, i)
                self.target_actor_map[key] = agent.target_actor

        self.online_actor_map = {}
        for brain_name, brain in self.brain_set:
            for i, agent in enumerate(brain.agents):
                key = "{}_{}".format(brain_name, i)
                self.online_actor_map[key] = agent.online_actor

        self.softmax = torch.nn.Softmax(dim=-1)

        self.huber_errors = torch.nn.SmoothL1Loss(reduction='none')

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    def step(self, episode_number: int):
        pass

    def step_episode(self, episode: int):
        """ Perform any end-of-episode updates """
        self.epsilon = self.epsilon_scheduler.get_param(episode)
        for brain_name, brain in self.brain_set:
            for agent in brain.agents:
                if hasattr(agent.online_actor, 'step_episode'):
                    agent.online_actor.step_episode()
                if hasattr(agent.online_critic, 'step_episode'):
                    agent.online_critic.step_episode()

    def get_action(self, state: torch.Tensor, online_actor: torch.nn.Module) -> Action:
        """Returns actions for given state as per current policy."""
        def get_actions_():
            online_actor.eval()
            with torch.no_grad():
                actions_ = online_actor(state)
            online_actor.train()
            return actions_

        if self.training:
            r = np.random.random()
            if r <= self.epsilon:
                action = self.random_action_generator.sample()
            else:
                action = get_actions_().cpu().data.numpy()
        else:
            action = get_actions_().cpu().data.numpy()

        if self.random_action_generator.continuous_actions:
            action = np.clip(
                action,
                self.random_action_generator.continuous_action_range[0],
                self.random_action_generator.continuous_action_range[1],
            )  # epsilon greedy policy
        action = Action(value=action)
        return action

    def get_random_action(self, *args) -> Action:
        """ Get a random action (used for warmup) """
        action = self.random_action_generator.sample()
        action = Action(value=action)
        return action

    def get_other_agent_atributes(self, x: torch.Tensor, agent_slicing_dict: dict, apply_fn_map: dict = None):
        output = []
        for k, f in agent_slicing_dict.items():
            if k != self.agent_id:
                if apply_fn_map:
                    output.append(apply_fn_map[k](f(x)))
                else:
                    output.append(f(x))

        output = torch.cat(output, dim=1)
        return output

    def get_agent_attributes(self, x: torch.Tensor, agent_slicing_dict: dict):
        return agent_slicing_dict[self.agent_id](x)

    def compute_actor_errors(self, experience_batch: ExperienceBatch, online_actor, online_critic, target_actor, target_critic, *args, **kwargs) -> tuple:
        """ Compute the error and loss of the actor"""
        other_agent_actions = self.get_other_agent_atributes(
            experience_batch.joint_states,
            self.map_agent_to_state_slice,
            apply_fn_map=self.online_actor_map
        ).detach()

        # Impose structure in the critic input to help with learning, of the in the form
        # [<agent_state> <other_agent_states>, <other_agent_actions>]
        agent_action = online_actor(experience_batch.states)

        other_agent_states = self.get_other_agent_atributes(
            experience_batch.joint_states, self.map_agent_to_state_slice
        )

        if self.matd3:
            actor_errors = - online_critic.qa(experience_batch.states, other_agent_states, other_agent_actions.float(), agent_action)
        else:
            actor_errors = - online_critic(experience_batch.states, other_agent_states, other_agent_actions.float(), agent_action)

        actor_loss = actor_errors.mean()
        return actor_loss, actor_errors

    def compute_critic_errors(self, experience_batch: ExperienceBatch, online_actor, online_critic, target_actor, target_critic, *args, **kwargs) -> tuple:
        """ Compute the error and loss of the critic"""

        if not self.matd3:
            other_agent_next_actions = self.get_other_agent_atributes(
                experience_batch.joint_next_states,
                self.map_agent_to_state_slice,
                apply_fn_map=self.target_actor_map
            )
            agent_next_actions = target_actor(experience_batch.next_states).float()
            other_agent_next_states_tensor = self.get_other_agent_atributes(experience_batch.joint_next_states,
                                                                            self.map_agent_to_state_slice)

            other_agent_actions = self.get_other_agent_atributes(
                experience_batch.joint_actions, self.map_agent_to_action_slice
            ).float()

            other_agent_states = self.get_other_agent_atributes(
                experience_batch.joint_states, self.map_agent_to_state_slice
            )
            with torch.no_grad():
                q_target_next = target_critic(
                    experience_batch.next_states, other_agent_next_states_tensor,
                    other_agent_next_actions.float(), agent_next_actions
                )

            q_targets = experience_batch.rewards + (self.gamma * q_target_next * (1 - experience_batch.dones))

            q_expected = online_critic(
                experience_batch.states,
                other_agent_states,
                other_agent_actions,
                experience_batch.actions.float()
            )

            # critic loss
            td_errors = self.huber_errors(q_expected, q_targets.detach())
            critic_loss = td_errors.mean()
            return critic_loss, td_errors
        else:
            other_agent_next_actions = self.get_other_agent_atributes(
                experience_batch.joint_next_states,
                self.map_agent_to_state_slice,
                apply_fn_map=self.target_actor_map
            )
            other_agent_next_states_tensor = self.get_other_agent_atributes(experience_batch.joint_next_states,
                                                                            self.map_agent_to_state_slice)

            other_agent_actions = self.get_other_agent_atributes(
                experience_batch.joint_actions, self.map_agent_to_action_slice
            ).float()

            other_agent_states = self.get_other_agent_atributes(
                experience_batch.joint_states, self.map_agent_to_state_slice
            )

            agent_next_actions = target_actor(experience_batch.next_states).float()

            if self.continuous_actions:
                agent_next_actions += self.gaussian_noise.sample(agent_next_actions).float().to(device)
                other_agent_next_actions += self.gaussian_noise.sample(other_agent_next_actions).float().to(device)

            if self.continuous_actions_clip_range is not None:
                agent_next_actions = agent_next_actions.clamp(
                    self.continuous_actions_clip_range[0], self.continuous_actions_clip_range[1]
                )
                other_agent_next_actions = other_agent_next_actions.clamp(
                    self.continuous_actions_clip_range[0], self.continuous_actions_clip_range[1]
                )

            with torch.no_grad():
                target_q1, target_q2 = target_critic(
                    experience_batch.next_states, other_agent_next_states_tensor,
                    other_agent_next_actions, agent_next_actions
                )

            min_target_q = torch.min(target_q1, target_q2)
            target_q = (experience_batch.rewards + (self.gamma * min_target_q * (1 - experience_batch.dones))).detach()

            current_q1, current_q2 = online_critic(
                experience_batch.states, other_agent_states,
                other_agent_actions, experience_batch.actions.float()
            )

            td_errors_1 = self.huber_errors(current_q1, target_q)
            td_errors_2 = self.huber_errors(current_q2, target_q)

            td_errors = td_errors_1 + td_errors_2
            critic_loss = td_errors.mean()
            return critic_loss, td_errors

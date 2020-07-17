import torch
import numpy as np
from typing import Optional, Tuple
from tools.misc import set_seed
from tools.rl_constants import ExperienceBatch, RandomBrainAction
from tools.parameter_decay import ParameterScheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGPolicy:
    """ Policy for the MADDPG algorithm"""
    def __init__(
            self,
            noise_factory,
            num_agents,
            action_dim: int,
            epsilon_scheduler: ParameterScheduler,
            continuous_actions: bool = True,
            gamma: float = 0.99,
            seed: Optional[int] = None,
            continuous_action_range: Tuple[float, float] = (-1, 1),
            discrete_action_range: Tuple[int, int] = (0, 1)
    ):

        if seed:
            self.set_seed(seed)
        self.gamma = gamma
        self.noise = noise_factory()
        self.num_agents = 4 #num_agents
        self.action_dim = action_dim

        self.epsilon_scheduler = epsilon_scheduler

        self.random_action_generator = RandomBrainAction(
            action_dim,
            num_agents,
            continuous_actions=continuous_actions,
            continuous_action_range=continuous_action_range,
            discrete_action_range=discrete_action_range
        )

        self.t_step = 0
        self.epsilon = self.epsilon_scheduler.initial

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    def step(self, episode_number: int):
        pass

    def step_episode(self, episode: int):
        """ Perform any end-of-episode updates """
        self.epsilon = self.epsilon_scheduler.get_param(episode)
        self.noise.reset()

    # def get_action(self, state: torch.Tensor, online_actor: torch.nn.Module, training: bool = True) -> np.ndarray:
    #     """Returns actions for given state as per current policy."""
    #     online_actor.eval()
    #     with torch.no_grad():
    #         action = online_actor(state).cpu().data.numpy()
    #     online_actor.train()
    #
    #     if training:
    #         action += self.noise.sample(action)
    #
    #     if self.random_action_generator.continuous_actions:
    #         action = np.clip(
    #             action,
    #             self.random_action_generator.continuous_action_range[0],
    #             self.random_action_generator.continuous_action_range[1],
    #         )  # epsilon greedy policy
    #     return action

    def get_action(self, state: torch.Tensor, online_actor: torch.nn.Module, training: bool = False) -> np.ndarray:
        """Returns actions for given state as per current policy."""
        # assert state.shape[0] == self.num_agents, state.shape[0]

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
        return action

    def get_random_action(self, *args) -> np.ndarray:
        """ Get a random action (used for warmup) """
        return self.random_action_generator.sample()

    def compute_actor_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic, agent_number) -> tuple:
        """ Compute the error and loss of the actor"""
        batch_size = len(experience_batch)
        action_pr_self = online_actor(experience_batch.states)

        expanded_joint_next_states = experience_batch.joint_states.view(batch_size * self.num_agents, -1).float()

        all_other_agent_states = torch.stack([row for i, row in enumerate(expanded_joint_next_states) if i % self.num_agents != 0]).float()

        action_pr_other = online_actor(all_other_agent_states).detach().float().view(batch_size, -1)

        critic_local_input2 = torch.cat((experience_batch.joint_states, action_pr_other), dim=1)
        actor_errors = -online_critic(critic_local_input2, action_pr_self)
        actor_loss = actor_errors.mean()
        return actor_loss, actor_errors

    def compute_critic_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic, agent_number) -> tuple:
        """ Compute the error and loss of the critic"""
        batch_size = len(experience_batch)

        # print(batch_size, self.num_agents)
        all_next_actions = target_actor(experience_batch.joint_next_states.view(batch_size * self.num_agents, -1).float())

        # Reshape as bsize, -1 as we can't be sure of the number of other agents
        all_other_agent_actions = torch.stack([row for i, row in enumerate(all_next_actions) if i % self.num_agents != 0]).view(batch_size, -1)
        all_agent_actions = all_next_actions.view(batch_size * self.num_agents, -1)[::self.num_agents]

        # print("all_other_agent_actions: {}".format(all_other_agent_actions.shape))


        critic_target_input = torch.cat((experience_batch.joint_next_states, all_other_agent_actions.float()), dim=1).to(
            device)

        # print("critic_target_input: {}".format(critic_target_input.shape))

        with torch.no_grad():
            q_target_next = target_critic(critic_target_input, all_agent_actions)
        q_targets = experience_batch.rewards + (self.gamma * q_target_next * (1 - experience_batch.dones))

        joint_actions = experience_batch.joint_actions.view(batch_size * self.num_agents, -1)
        all_other_agent_actions = torch.stack([row for i, row in enumerate(joint_actions) if i % self.num_agents != 0]).view(batch_size, -1)

        critic_local_input = torch.cat((experience_batch.joint_states, all_other_agent_actions.float()), dim=1).to(device)
        q_expected = online_critic(critic_local_input, experience_batch.actions.float())

        # critic loss
        huber_errors = torch.nn.SmoothL1Loss(reduction='none')
        td_errors = huber_errors(q_expected, q_targets.detach())
        critic_loss = td_errors.mean()

        return critic_loss, td_errors

    # def compute_actor_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic, agent_number) -> tuple:
    #     """ Compute the error and loss of the actor"""
    #     batch_size = len(experience_batch)
    #     actions = online_actor(experience_batch.joint_states.view(batch_size * 2, -1)).view(batch_size, -1)
    #
    #     # expanded_joint_next_states = experience_batch.joint_next_states.view(batch_size * self.num_agents, -1).float()
    #
    #     # all_other_agent_states = torch.stack([row for i, row in enumerate(expanded_joint_next_states) if i % self.num_agents != 0]).float()
    #     #
    #     # action_pr_other = online_actor(all_other_agent_states).detach().float()
    #     #
    #     # critic_local_input2 = torch.cat((experience_batch.joint_states, action_pr_other), dim=1)
    #     actor_errors = -online_critic(experience_batch.joint_states, actions)
    #     actor_loss = actor_errors.mean()
    #     return actor_loss, actor_errors
    #
    # def compute_critic_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic, agent_number) -> tuple:
    #     """ Compute the error and loss of the critic"""
    #     batch_size = len(experience_batch)
    #
    #     all_next_actions = target_actor(experience_batch.joint_next_states.view(batch_size * self.num_agents, -1).float())
    #
    #     # all_other_agent_actions = torch.stack([action for i, action in enumerate(all_next_actions) if i % self.num_agents != 0])
    #     # all_agent_actions = all_next_actions.view(batch_size * self.num_agents, -1)[::self.num_agents]
    #
    #     # critic_target_input = torch.cat((experience_batch.joint_next_states, all_other_agent_actions.float()), dim=1).to(
    #     #     device)
    #
    #     with torch.no_grad():
    #         q_target_next = target_critic(experience_batch.joint_next_states, all_next_actions.view(batch_size, -1))
    #     q_targets = experience_batch.rewards + (self.gamma * q_target_next * (1 - experience_batch.dones))
    #
    #     joint_actions = experience_batch.joint_actions.view(batch_size * self.num_agents, -1)
    #     # all_other_agent_actions = torch.stack([row for i, row in enumerate(joint_actions) if i % self.num_agents != 0])
    #     # critic_local_input = torch.cat((experience_batch.joint_states, all_other_agent_actions.float()), dim=1).to(device)
    #     q_expected = online_critic(experience_batch.joint_states, joint_actions.view(batch_size, -1))
    #
    #     # critic loss
    #     huber_errors = torch.nn.SmoothL1Loss(reduction='none')
    #     td_errors = huber_errors(q_expected, q_targets.detach())
    #     critic_loss = td_errors.mean()
    #
    #     return critic_loss, td_errors

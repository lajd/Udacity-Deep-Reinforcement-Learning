import torch
import numpy as np
from typing import Optional, Tuple
from tools.misc import set_seed
from tools.rl_constants import ExperienceBatch, RandomBrainAction
from tools.parameter_decay import ParameterScheduler
import torch.nn.functional as F
from conf import TEST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGPolicy:
    """ Policy for the MADDPG algorithm"""
    def __init__(
            self,
            noise_factory,
            num_agents,
            action_dim: int,
            continuous_actions: bool = True,
            gamma: float = 0.99,
            seed: Optional[int] = None,
            continuous_action_range: Tuple[float, float] = (-1, 1),
            discrete_action_range: Tuple[int, int] = (0, 1)
    ):

        if seed:
            self.set_seed(seed)
        self.gamma = gamma
        self.agent_noise = [noise_factory() for _ in range(num_agents)]
        self.num_agents = num_agents
        # self.noise = noise
        self.action_dim = action_dim

        self.random_action_generator = RandomBrainAction(
            action_dim,
            num_agents,
            continuous_actions=continuous_actions,
            continuous_action_range=continuous_action_range,
            discrete_action_range=discrete_action_range
        )

        self.t_step = 0
        self.n_step = 0

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    def step_episode(self, episode: int):
        """ Perform any end-of-episode updates """
        for i in range(self.num_agents):
            self.agent_noise[i].reset()

    def get_action(self, state: torch.Tensor, online_actor: torch.nn.Module, with_noise: bool = True, training: bool = False) -> np.ndarray:
        """Returns actions for given state as per current policy."""
        assert state.shape[0] == self.num_agents, state.shape[0]

        self.n_step += 1
        epsilon = max((1500 - self.n_step) / 1500, .01)

        online_actor.eval()
        with torch.no_grad():
            actions = online_actor(state)
        online_actor.train()

        if training:
            r = np.random.random()
            if r <= epsilon:
                action = np.random.uniform(-1, 1, (self.num_agents, self.action_dim))
            else:
                action = np.clip(actions.cpu().data.numpy(), -1, 1)  # epsilon greedy policy
        else:
            action = actions.cpu().data.numpy()
        return action

    def get_random_action(self, *args) -> np.ndarray:
        """ Get a random action (used for warmup) """
        return self.random_action_generator.sample()

    def compute_actor_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic, agent_number) -> tuple:
        """ Compute the error and loss of the actor"""
        batch_size = len(experience_batch)
        action_pr_self = online_actor(experience_batch.states)

        # if TEST:
        #     joint_next_states = experience_batch.joint_next_states.view(batch_size, self.num_agents, -1)
        #     all_other_agent_joint_next_states = torch.stack([joint_next_states[:, i, :] for i in range(self.num_agents) if i != agent_number]).view(batch_size, -1)
        #
        #     all_other_agent_joint_next_states = torch.stack([joint_next_states[:, i, :] for i in range(self.num_agents) if i != agent_number]).view(batch_size, -1)
        #     action_pr_other = online_actor(all_other_agent_joint_next_states).detach()
        # else:

        expanded_joint_next_states = experience_batch.joint_next_states.view(batch_size * self.num_agents, -1)

        all_other_agent_states = torch.stack([row for i, row in enumerate(expanded_joint_next_states) if i % self.num_agents != 0])

        action_pr_other = online_actor(all_other_agent_states).detach()

        critic_local_input2 = torch.cat((experience_batch.joint_states, action_pr_other), dim=1)
        actor_errors = -online_critic(critic_local_input2, action_pr_self)
        actor_loss = actor_errors.mean()
        return actor_loss, actor_errors

    def compute_critic_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic, agent_number) -> tuple:
        """ Compute the error and loss of the critic"""
        batch_size = len(experience_batch)

        # if TEST:
        #     all_next_actions = target_actor(
        #         experience_batch.joint_next_states.view(
        #             batch_size * self.num_agents, -1
        #         )
        #     )
        #
        #     # Get all the actions done by all other agents of shape (batch_size, (num_agents - 1) * agent_action_size)
        #     all_next_actions = all_next_actions.view(batch_size, self.num_agents, -1)
        #     all_other_agent_actions = torch.stack([all_next_actions[:, i, :] for i in range(self.num_agents) if i != agent_number]).view(batch_size, -1)
        #
        #     critic_target_input = torch.cat((experience_batch.joint_next_states, all_other_agent_actions), dim=1).to(device)
        #
        #     all_agent_actions = all_next_actions[:, agent_number, :].view(batch_size, -1)
        #
        #     with torch.no_grad():
        #         q_target_next = target_critic(critic_target_input, all_agent_actions)
        #     q_targets = experience_batch.rewards + (self.gamma * q_target_next * (1 - experience_batch.dones))
        #
        #     all_next_states = experience_batch.joint_actions.view(batch_size, self.num_agents, -1)
        #     all_other_agent_states = torch.stack([all_next_states[:, i, :] for i in range(self.num_agents) if i != agent_number]).view(batch_size, -1)
        #
        #     critic_local_input = torch.cat((experience_batch.joint_states, all_other_agent_states), dim=1).to(device)
        #     q_expected = online_critic(critic_local_input, experience_batch.actions)
        #
        # else:

        all_next_actions = target_actor(experience_batch.joint_next_states.view(batch_size * self.num_agents, -1).float())

        # all_other_agent_actions = all_next_actions.view(batch_size * self.num_agents, -1)[1::self.num_agents]
        # all_other_agent_actions = torch.stack([all_next_actions[:, i, :] for i in range(self.num_agents) if i != agent_number]).view(batch_size, -1)
        all_other_agent_actions = torch.stack([row for i, row in enumerate(all_next_actions) if i % self.num_agents != 0])
        all_agent_actions = all_next_actions.view(batch_size * self.num_agents, -1)[::self.num_agents]

        critic_target_input = torch.cat((experience_batch.joint_next_states, all_other_agent_actions.float()), dim=1).to(
            device)

        with torch.no_grad():
            q_target_next = target_critic(critic_target_input, all_agent_actions)
        q_targets = experience_batch.rewards + (self.gamma * q_target_next * (1 - experience_batch.dones))

        joint_actions = experience_batch.joint_actions.view(batch_size * self.num_agents, -1)
        all_other_agent_actions = torch.stack([row for i, row in enumerate(joint_actions) if i % self.num_agents != 0])
        critic_local_input = torch.cat((experience_batch.joint_states, all_other_agent_actions.float()), dim=1).to(device)
        q_expected = online_critic(critic_local_input, experience_batch.actions.float())

        # critic loss
        huber_errors = torch.nn.SmoothL1Loss(reduction='none')
        td_errors = huber_errors(q_expected, q_targets.detach())
        critic_loss = td_errors.mean()

        return critic_loss, td_errors

    # def test(self):
    #
    #     # need to transpose each element of the samples
    #     # to flip obs[parallel_agent][agent_number] to
    #     # obs[agent_number][parallel_agent]
    #     obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
    #
    #     obs_full = torch.stack(obs_full)
    #     next_obs_full = torch.stack(next_obs_full)
    #
    #     agent = self.maddpg_agent[agent_number]
    #     agent.critic_optimizer.zero_grad()
    #
    #     # critic loss = batch mean of (y- Q(s,a) from target network)^2
    #     # y = reward of this timestep + discount * Q(st+1,at+1) from target network
    #     print(next_obs.shape)
    #     target_actions = self.target_act(next_obs)
    #     target_actions = torch.cat(target_actions, dim=1)
    #
    #     target_critic_input = torch.cat((next_obs_full.t(), target_actions), dim=1).to(device)
    #
    #     with torch.no_grad():
    #         q_next = agent.target_critic(target_critic_input)
    #
    #     y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
    #     action = torch.cat(action, dim=1)
    #     critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
    #     q = agent.critic(critic_input)
    #
    #     huber_loss = torch.nn.SmoothL1Loss()
    #     critic_loss = huber_loss(q, y.detach())
    #     critic_loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
    #     agent.critic_optimizer.step()
    #
    #     # update actor network using policy gradient
    #     agent.actor_optimizer.zero_grad()
    #     # make input to agent
    #     # detach the other agents to save computation
    #     # saves some time for computing derivative
    #     q_input = [self.maddpg_agent[i].actor(ob) if i == agent_number \
    #                    else self.maddpg_agent[i].actor(ob).detach()
    #                for i, ob in enumerate(obs)]
    #
    #     q_input = torch.cat(q_input, dim=1)
    #     # combine all the actions and observations for input to critic
    #     # many of the obs are redundant, and obs[1] contains all useful information already
    #     q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
    #
    #     # get the policy gradient
    #     actor_loss = -agent.critic(q_input2).mean()
    #     actor_loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
    #     agent.actor_optimizer.step()
    #
    #     al = actor_loss.cpu().detach().item()
    #     cl = critic_loss.cpu().detach().item()

    def get_agent_data(self, x: torch.Tensor):
        pass

    def get_other_agent_data(self, x: torch.Tensor):
        pass

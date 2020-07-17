from typing import Callable
from agents.base import Agent
import random
import numpy as np
import torch
from tools.misc import soft_update
from tools.rl_constants import Experience, ExperienceBatch
from agents.ddpg_agent import DDPGAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class IndependentMADDPGAgent(Agent):
    """Interacts with and learns from the environment."""
    def __init__(
            self,
            state_shape,
            action_size,
            seed,
            agents,
            memory_factory: Callable,
            num_learning_updates=10,
            tau: float = 1e-2, batch_size: int = 512, update_frequency: int = 20,
            critic_grad_norm_clip: int = 1, policy_update_frequency: int = 2
    ):
        """Initialize an Agent object.

        Params
        ======
            state_shape (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(action_size=action_size, state_shape=state_shape, num_agents=len(agents))
        self.seed = random.seed(seed)
        self.n_seed = np.random.seed(seed)

        self.num_learning_updates = num_learning_updates
        self.tau = tau

        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.critic_grad_norm_clip = critic_grad_norm_clip
        self.policy_update_frequency = policy_update_frequency

        self.agents = agents

        # self.policy = policy
        #
        # # critic local and target network (Q-Learning)
        # self.online_critic = critic_factory().to(device).float()
        # self.target_critic = critic_factory().to(device).float()
        # self.target_critic.load_state_dict(self.online_critic.state_dict())
        #
        # # actor local and target network (Policy gradient)
        # self.online_actor = actor_factory().to(device).float()
        # self.target_actor = actor_factory().to(device).float()
        # self.target_actor.load_state_dict(self.online_actor.state_dict())
        #
        # # optimizer for critic and actor network
        # self.critic_optimizer = critic_optimizer_factory(self.online_critic.parameters())
        # self.actor_optimizer = actor_optimizer_factory(self.online_actor.parameters())
        # Replay memory
        self.memory = memory_factory()

    def set_mode(self, mode: str):
        for agent in self.agents:
            agent.set_mode(mode)

    def step(self, experience: Experience, **kwargs):
        self.memory.add(experience)

        if self.warmup:
            return
        else:
            self.t_step += 1
            if self.t_step % self.update_frequency == 0 and len(self.memory) > self.batch_size:
                # If enough samples are available in memory, get random subset and learn
                for i in range(self.num_learning_updates):
                    experiences = self.memory.sample(self.batch_size)
                    self.learn(experiences)

    def step_episode(self, i_episode):
        # Reset the noise modules
        for agent in self.agents:
            agent.policy.step_episode(i_episode)

    def get_action(self, state, training=True) -> np.ndarray:
        return self.policy.get_action(state, self.online_actor, training=training)

    def get_random_action(self, *args) -> np.ndarray:
        """ Get a random action, used for warmup"""
        return self.policy.get_random_action()

    def learn(self, experience_batch: ExperienceBatch, agent_number: int):

        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        experience_batch = experience_batch.to(device)
        critic_loss, critic_errors = self.policy.compute_critic_errors(
            experience_batch,
            online_actor=self.online_actor,
            online_critic=self.online_critic,
            target_actor=self.target_actor,
            target_critic=self.target_critic,
            agent_number=agent_number
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_critic.parameters(), self.critic_grad_norm_clip)
        self.critic_optimizer.step()

        if self.t_step % self.policy_update_frequency == 0:
            # Delay the policy update as in TD3
            actor_loss, actor_errors = self.policy.compute_actor_errors(
                experience_batch,
                online_actor=self.online_actor,
                online_critic=self.online_critic,
                target_actor=self.target_actor,
                target_critic=self.target_critic,
                agent_number=agent_number
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            soft_update(self.online_critic, self.target_critic, self.tau)
            soft_update(self.online_actor, self.target_actor, self.tau)
            return critic_loss, critic_errors, actor_loss, actor_errors
        return critic_loss, critic_errors, None, None

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        print(next_obs.shape)
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)

        target_critic_input = torch.cat((next_obs_full.t(), target_actions), dim=1).to(device)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [self.maddpg_agent[i].actor(ob) if i == agent_number \
                       else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs)]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

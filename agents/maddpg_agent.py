from typing import Callable
from agents.base import Agent
from tools.misc import *
from tools.rl_constants import Experience, ExperienceBatch, Action
from tools.misc import set_seed
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.autograd.set_detect_anomaly(True)


class MADDPGAgent(Agent):
    """Interacts with and learns from the environment."""
    online_critic = None
    online_actor = None
    target_critic = None
    target_actor = None
    memory = None
    actor_optimizer = None
    critic_optimizer = None

    def __init__(
            self,
            agent_id,
            policy,
            state_shape,
            action_size,
            seed,
            critic_factory: Callable,
            actor_factory: Callable,
            critic_optimizer_factory: Callable,
            actor_optimizer_factory: Callable,
            memory_factory: Callable,
            num_learning_updates=10,
            tau: float = 1e-2, batch_size: int = 512, update_frequency: int = 20,
            critic_grad_norm_clip: int = 1, policy_update_frequency: int = 2,
            homogeneous_agents: bool = False
        ):

        super().__init__(action_size=action_size, state_shape=state_shape)
        if seed is not None:
            set_seed(seed)
        self.n_seed = np.random.seed(seed)
        self.num_learning_updates = num_learning_updates
        self.tau = tau
        self.agent_id = agent_id

        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.critic_grad_norm_clip = critic_grad_norm_clip
        self.policy_update_frequency = policy_update_frequency

        self.policy = policy

        self.homogeneous_agents = homogeneous_agents

        # critic local and target network (Q-Learning)
        if self.homogeneous_agents and MADDPGAgent.online_critic is None:
            MADDPGAgent.online_critic = critic_factory().to(device).float()

            MADDPGAgent.target_critic = critic_factory().to(device).float()
            MADDPGAgent.target_critic.load_state_dict(self.online_critic.state_dict())

            # actor local and target network (Policy gradient)
            MADDPGAgent.online_actor = actor_factory().to(device).float()
            MADDPGAgent.target_actor = actor_factory().to(device).float()
            MADDPGAgent.target_actor.load_state_dict(self.online_actor.state_dict())

            # optimizer for critic and actor network
            MADDPGAgent.critic_optimizer = critic_optimizer_factory(self.online_critic.parameters())
            MADDPGAgent.actor_optimizer = actor_optimizer_factory(self.online_actor.parameters())

            self.online_critic = MADDPGAgent.online_critic
            self.target_critic = MADDPGAgent.target_critic

            # actor local and target network (Policy gradient)
            self.online_actor = MADDPGAgent.online_actor
            self.target_actor = MADDPGAgent.target_actor

            # optimizer for critic and actor network
            self.critic_optimizer = MADDPGAgent.critic_optimizer
            self.actor_optimizer = MADDPGAgent.actor_optimizer
        else:
            self.online_critic = critic_factory().to(device).float()
            self.target_critic = critic_factory().to(device).float()
            self.target_critic.load_state_dict(self.online_critic.state_dict())

            # actor local and target network (Policy gradient)
            self.online_actor = actor_factory().to(device).float()
            self.target_actor = actor_factory().to(device).float()
            self.target_actor.load_state_dict(self.online_actor.state_dict())

            # optimizer for critic and actor network
            self.critic_optimizer = critic_optimizer_factory(self.online_critic.parameters())
            self.actor_optimizer = actor_optimizer_factory(self.online_actor.parameters())
        self.memory = memory_factory()

    def set_mode(self, mode: str):
        if mode == 'train':
            self.online_actor.train()
            self.online_critic.train()
            self.policy.train()
        elif mode == 'eval':
            self.online_actor.eval()
            self.online_critic.eval()
            self.policy.eval()
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

    def step(self, experience: Experience, **kwargs):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(experience)

        if self.warmup:
            return
        else:
            self.t_step += 1
            if self.t_step % self.update_frequency == 0 and len(self.memory) > self.batch_size:
                # If enough samples are available in memory, get random subset and learn
                for i in range(self.num_learning_updates):
                    experience_batch = self.memory.sample(self.batch_size)
                    experience_batch = experience_batch.to(device)

                    critic_loss, critic_errors, actor_loss, actor_errors = self.learn(experience_batch)

                    with torch.no_grad():
                        if critic_errors.min() < 0:
                            raise RuntimeError("Errors must be > 0, found {}".format(critic_errors.min()))

                        priorities = critic_errors.detach().cpu().numpy()
                        self.memory.update(experience_batch.sample_idxs, priorities)

                    self.param_capture.add('critic_loss', critic_loss)
                    self.param_capture.add('actor_loss', actor_loss)

    def step_episode(self, i_episode, *args):
        # Reset the noise modules
        self.policy.step_episode(i_episode)

    def get_action(self, state, *args, **kwargs) -> Action:
        state = state.to(device)
        action: Action = self.policy.get_action(state, self.online_actor)
        return action

    def get_random_action(self, *args, **kwargs) -> Action:
        """ Get a random action, used for warmup"""
        action: Action = self.policy.get_random_action()
        return action

    def learn(self, experience_batch: ExperienceBatch) -> tuple:
        """Update value parameters using given batch of experience tuples and return TD error

        Args:
            experience_batch (ExperienceBatch): Batch of experiences

        Returns:
            critic_loss (torch.FloatTensor): The TD errors for each sample
            critic_errors
            actor_loss
            actor_errors
        """
        experience_batch = experience_batch.to(device)
        critic_loss, critic_errors = self.policy.compute_critic_errors(
            experience_batch,
            online_actor=self.online_actor,
            online_critic=self.online_critic,
            target_actor=self.target_actor,
            target_critic=self.target_critic,
            agent_num=self.agent_id
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
                agent_num=self.agent_id
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            soft_update(self.online_critic, self.target_critic, self.tau)
            soft_update(self.online_actor, self.target_actor, self.tau)
            return critic_loss, critic_errors, actor_loss, actor_errors
        return critic_loss, critic_errors, None, None


class DummyMADDPGAgent(Agent):
    """Interacts with and learns from the environment."""
    def __init__(self, state_shape, action_size, seed, map_agent_to_state_slice, map_agent_to_action_slice):
        """Initialize an Agent object.

        Params
        ======
            state_shape (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(action_size=action_size, state_shape=state_shape)
        if seed is not None:
            set_seed(seed)
        self.target_actor = lambda x: torch.randint(0, self.action_size + 1, (len(x), 1)).to(device)
        self.online_actor = lambda x: torch.randint(0, self.action_size + 1, (len(x), 1)).to(device)
        self.online_critic = {}
        self.map_agent_to_state_slice = map_agent_to_state_slice
        self.map_agent_to_action_slice = map_agent_to_action_slice

    def set_mode(self, mode: str):
        pass

    def step(self, experience: Experience, **kwargs):
        pass

    def step_episode(self, i_episode, *args):
        pass

    def get_action(self, state, training=True, *args, **kwargs) -> Action:
        # action = torch.rand(1, self.action_size)
        action = torch.randint(0, self.action_size + 1, (1, 1))
        action = action.cpu().data.numpy()
        return Action(value=action)

    def get_random_action(self, *args, **kwargs) -> Action:
        # action = torch.rand(1, self.action_size)
        action = torch.randint(0, self.action_size, (1, 1))
        action = action.cpu().data.numpy()
        return Action(value=action)

    def learn(self, experience_batch: ExperienceBatch) -> tuple:
        """Update value parameters using given batch of experience tuples and return TD error

        Args:
            experience_batch (ExperienceBatch): Batch of experiences

        Returns:
            critic_loss (torch.FloatTensor): The TD errors for each sample
            critic_errors
            actor_loss
            actor_errors
        """
        return [], [], None, None


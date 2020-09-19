import numpy as np
import random
from typing import Callable, Union, Tuple, Optional
from agents.memory.memory import Memory
from agents.memory.prioritized_memory import PrioritizedMemory
from tools.rl_constants import Experience, ExperienceBatch, Action
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from tools.misc import soft_update
import torch
from agents.base import Agent
from agents.policies.base_policy import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class DDPGAgent(Agent):
    """Interacts with and learns from the environment."""
    memory = None
    online_actor = None
    target_actor = None
    actor_optimizer = None

    online_critic = None
    target_critic = None
    critic_optimizer = None

    policy = None

    actor_optimizer_scheduler = None
    critic_optimizer_scheduler = None

    episode_counter = 0

    def __init__(
            self,
            state_shape: int,
            action_size: int,
            random_seed: int,
            memory_factory: Callable[[], Union[Memory, PrioritizedMemory]],
            actor_model_factory: Callable[[], torch.nn.Module],
            actor_optimizer_factory: Callable,
            actor_optimizer_scheduler: Callable[[Optimizer], _LRScheduler],
            critic_model_factory: Callable[[], torch.nn.Module],
            critic_optimizer_factory: Callable,
            critic_optimizer_scheduler: Callable[[Optimizer], _LRScheduler],
            policy_factory: Callable[[], Policy],
            agent_id: Optional[str] = None,
            update_frequency: int = 20,
            n_learning_iterations: int = 10,
            batch_size: int = 512,
            gamma: float = 0.99,
            tau: float = 1e-2,
            policy_update_frequency: int = 2,
            critic_grad_norm_clip: float = 1,
            td3: bool = False,
            shared_agent_brain: bool = False
    ):
        """Initialize an Agent object.
        Params
        ======
            state_shape (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            memory_factory: (Callable) Return a Memory or PrioritizedMemory object
            actor_model_factory: (Callable) Return an actor instance subclassing torch.nn.Module
            actor_optimizer_factory: (callable) Return an optimizer for the actor
            actor_optimizer_scheduler: (callable) Return a subclass of _LRScheduler for scheduling actor optimizer LR
            critic_model_factory: (Callable) Return an critic instance subclassing torch.nn.Module
            critic_optimizer_factory: (callable) Return an optimizer for the critic
            critic_optimizer_scheduler: (callable) Return a subclass of _LRScheduler for scheduling actor optimizer LR
            policy_factory: (callable): Return a policy
            update_frequency: (int) Frequency at which to update the policy network and targets
            n_learning_iterations: (int) Number of learning iterations to perform at each update step
            batch_size: (int) Minibatch size
            gamma: (float) Discount factor
            tau: (float) Parameter used for soft copying; tau=1 -> a hard copy
            policy_update_frequency (int, default=1): The number of time steps to wait before optimizing the policy &
                updating the target networks. Introduced in TD3.
            shared_agent_brain (bool): Use a shared brain/model/optimizer for all agents
        """
        super().__init__(action_size=action_size, state_shape=state_shape)

        self.agent_id = agent_id
        self.shared_agent_brain = shared_agent_brain
        if not self.shared_agent_brain:

            # Shared Memory
            DDPGAgent.memory = memory_factory()

            DDPGAgent.online_actor = actor_model_factory().to(device).float().train()
            DDPGAgent.target_actor = actor_model_factory().to(device).float().eval()
            DDPGAgent.target_actor.load_state_dict(DDPGAgent.online_actor.state_dict())

            # Shared Critic network
            DDPGAgent.online_critic = critic_model_factory().to(device).float().train()
            DDPGAgent.target_critic = critic_model_factory().to(device).float().eval()
            DDPGAgent.target_critic.load_state_dict(DDPGAgent.online_critic.state_dict())

            DDPGAgent.actor_optimizer = actor_optimizer_factory(DDPGAgent.online_actor.parameters())
            DDPGAgent.actor_optimizer_scheduler = actor_optimizer_scheduler(DDPGAgent.actor_optimizer)

            DDPGAgent.critic_optimizer = critic_optimizer_factory(DDPGAgent.online_critic.parameters())
            DDPGAgent.critic_optimizer_scheduler = critic_optimizer_scheduler(DDPGAgent.actor_optimizer)

            # Shared Policy
            DDPGAgent.policy = policy_factory()
        else:
            if DDPGAgent.memory is None:
                # Shared Memory
                DDPGAgent.memory = memory_factory()

            # Shared Actor network
            if DDPGAgent.online_actor is None:
                DDPGAgent.online_actor = actor_model_factory().to(device).train()
            if DDPGAgent.target_actor is None:
                DDPGAgent.target_actor = actor_model_factory().to(device).eval()
            if DDPGAgent.actor_optimizer is None:
                DDPGAgent.actor_optimizer = actor_optimizer_factory(DDPGAgent.online_actor.parameters())
            if DDPGAgent.actor_optimizer_scheduler is None:
                DDPGAgent.actor_optimizer_scheduler = actor_optimizer_scheduler(DDPGAgent.actor_optimizer)

            # Shared Critic network
            if DDPGAgent.online_critic is None:
                DDPGAgent.online_critic = critic_model_factory().to(device).train()
            if DDPGAgent.target_critic is None:
                DDPGAgent.target_critic = critic_model_factory().to(device).eval()
            if DDPGAgent.critic_optimizer is None:
                DDPGAgent.critic_optimizer = critic_optimizer_factory(DDPGAgent.online_critic.parameters())
            if DDPGAgent.critic_optimizer_scheduler is None:
                DDPGAgent.critic_optimizer_scheduler = critic_optimizer_scheduler(DDPGAgent.actor_optimizer)

            self.policy = policy_factory()

        assert batch_size < DDPGAgent.memory.capacity, \
            "Batch size {} must be less than memory capacity {}".format(batch_size, DDPGAgent.memory.capacity)

        # Parameters
        self.seed = random.seed(random_seed)
        self.update_frequency = update_frequency
        self.n_learning_iterations = n_learning_iterations
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_update_frequency = policy_update_frequency
        self.critic_grad_norm_clip = critic_grad_norm_clip
        self.td3 = td3

    def set_mode(self, mode: str):
        if mode == 'train':
            DDPGAgent.online_actor.train()
            DDPGAgent.online_critic.train()
            self.policy.train()
        elif mode == 'eval':
            DDPGAgent.online_actor.eval()
            DDPGAgent.online_critic.eval()
            self.policy.eval()
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

    def step(self, experience: Experience, **kwargs):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        DDPGAgent.memory.add(experience)

        if self.warmup:
            return
        else:
            self.t_step += 1
            # Learn, if enough samples are available in memory
            if self.t_step % self.update_frequency == 0 and len(DDPGAgent.memory) > self.batch_size:
                for i in range(self.n_learning_iterations):
                    experience_batch: ExperienceBatch = DDPGAgent.memory.sample(self.batch_size)
                    critic_loss, critic_errors, actor_loss, actor_errors = self.learn(experience_batch)

                    # Update the priority replay buffer
                    with torch.no_grad():
                        if critic_errors.min() < 0:
                            raise RuntimeError("Errors must be > 0, found {}".format(critic_errors.min()))

                        priorities = critic_errors.detach().cpu().numpy()
                        DDPGAgent.memory.update(experience_batch.sample_idxs, priorities)

    def step_episode(self, episode: int,  *args) -> None:
        self.policy.step_episode(episode)

    def get_action(self, state: torch.Tensor, *args, **kwargs) -> Action:
        """Returns actions for given state as per current policy."""
        state = state.to(device)
        action: Action = self.policy.get_action(state, DDPGAgent.online_actor)
        return action

    def get_random_action(self, *args,**kwargs) -> Action:
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
            online_actor=DDPGAgent.online_actor,
            online_critic=DDPGAgent.online_critic,
            target_actor=DDPGAgent.target_actor,
            target_critic=DDPGAgent.target_critic,
        )

        DDPGAgent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(DDPGAgent.online_critic.parameters(), self.critic_grad_norm_clip)
        DDPGAgent.critic_optimizer.step()

        if self.t_step % self.policy_update_frequency == 0:
            # Delay the policy update as in TD3
            actor_loss, actor_errors = self.policy.compute_actor_errors(
                experience_batch,
                online_actor=DDPGAgent.online_actor,
                online_critic=DDPGAgent.online_critic,
                target_actor=DDPGAgent.target_actor,
                target_critic=DDPGAgent.target_critic,
            )

            DDPGAgent.actor_optimizer.zero_grad()
            actor_loss.backward()
            DDPGAgent.actor_optimizer.step()

            # Update target networks
            soft_update(DDPGAgent.online_critic, DDPGAgent.target_critic, self.tau)
            soft_update(DDPGAgent.online_actor, DDPGAgent.target_actor, self.tau)
            return critic_loss, critic_errors, actor_loss, actor_errors
        return critic_loss, critic_errors, None, None

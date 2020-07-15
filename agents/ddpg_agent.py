import numpy as np
import random
from typing import Callable, Union, Tuple
from agents.memory.memory import Memory
from agents.memory.prioritized_memory import PrioritizedMemory
from tools.rl_constants import Experience, ExperienceBatch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from tools.misc import soft_update
import torch
from agents.base import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.autograd.set_detect_anomaly(True)


class DDPGAgent(Agent):
    """Interacts with and learns from the environment."""
    def __init__(
            self,
            state_shape: int,
            action_size: int,
            random_seed: int,
            num_agents: int,
            memory_factory: Callable[[], Union[Memory, PrioritizedMemory]],
            actor_model_factory: Callable[[], torch.nn.Module],
            actor_optimizer_factory: Callable[[torch.nn.Module], Optimizer],
            actor_optimizer_scheduler: Callable[[Optimizer], _LRScheduler],
            critic_model_factory: Callable[[], torch.nn.Module],
            critic_optimizer_factory: Callable[[torch.nn.Module], Optimizer],
            critic_optimizer_scheduler: Callable[[Optimizer], _LRScheduler],
            policy_factory: Callable,
            update_frequency: int = 20,
            n_learning_iterations: int = 10,
            batch_size: int = 32,
            gamma: float = 0.99,
            tau: float = 1e-3,
            policy_update_frequency: int = 1,
            critic_grad_norm_clip: float = 1,
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
        """
        super().__init__(action_size=action_size, state_shape=state_shape, num_agents=num_agents)
        # Shared Memory
        self.memory = memory_factory()
        assert batch_size < self.memory.capacity, \
            "Batch size {} must be less than memory capacity {}".format(batch_size, self.memory.capacity)

        self.online_actor = actor_model_factory().to(device).train()
        self.target_actor = actor_model_factory().to(device).eval()
        self.actor_optimizer = actor_optimizer_factory(self.online_actor)
        self.actor_optimizer_scheduler = actor_optimizer_scheduler(self.actor_optimizer)

        # Shared Critic network
        self.online_critic = critic_model_factory().to(device).train()
        self.target_critic = critic_model_factory().to(device).eval()
        self.critic_optimizer = critic_optimizer_factory(self.online_critic)
        self.critic_optimizer_scheduler = critic_optimizer_scheduler(self.actor_optimizer)

        # Shared Policy
        self.policy = policy_factory()
        self.policy.step_episode(None)

        # Parameters
        self.seed = random.seed(random_seed)
        self.update_frequency = update_frequency
        self.n_learning_iterations = n_learning_iterations
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_update_frequency = policy_update_frequency
        self.critic_grad_norm_clip = critic_grad_norm_clip

    def set_mode(self, mode: str):
        if mode == 'train':
            self.online_actor.train()
            self.online_critic.train()
        elif mode == 'eval':
            self.online_actor.eval()
            self.online_critic.eval()
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

    def step_episode(self, episode: int):
        """ Perform any end-of-epsiode updates """
        self.policy.step_episode(episode)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.memory.step_episode(episode)
        self.episode_counter += 1

    def step(self, experience: Experience, **kwargs):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        self.memory.add(experience)

        if self.warmup:
            return
        else:
            self.t_step += 1
            # Learn, if enough samples are available in memory
            if self.t_step % self.update_frequency == 0 and len(self.memory) > self.batch_size:
                for i in range(self.n_learning_iterations):
                    experience_batch: ExperienceBatch = self.memory.sample(self.batch_size)
                    critic_loss, critic_errors, actor_loss, actor_errors = self.learn(experience_batch)

                    # Update the priority replay buffer
                    with torch.no_grad():
                        if critic_errors.min() < 0:
                            raise RuntimeError("Errors must be > 0, found {}".format(critic_errors.min()))

                        priorities = critic_errors.detach().cpu().numpy()
                        self.memory.update(experience_batch.sample_idxs, priorities)

    def get_action(self, state: torch.Tensor, add_noise=True) -> np.ndarray:
        """Returns actions for given state as per current policy."""
        state = state.to(device)
        action = self.policy.get_action(state, self.online_actor, add_noise)
        return action

    def get_random_action(self, *args):
        """ Get a random action, used for warmup"""
        return self.policy.get_random_action()

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
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            soft_update(self.online_critic, self.target_critic, self.tau)
            soft_update(self.online_actor, self.target_actor, self.tau)
            return critic_loss, critic_errors, actor_loss, actor_errors
        return critic_loss, critic_errors, None, None

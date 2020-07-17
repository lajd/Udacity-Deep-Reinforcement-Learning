from typing import Callable
from agents.base import Agent
from tools.misc import *
from tools.rl_constants import Experience, ExperienceBatch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class HomogeneousMADDPGAgent(Agent):
    """Interacts with and learns from the environment."""
    def __init__(self, policy, state_shape, action_size, num_agents, seed,
                 critic_factory: Callable,
                 actor_factory: Callable,
                 critic_optimizer_factory: Callable,
                 actor_optimizer_factory: Callable,
                 memory_factory: Callable,
                 num_learning_updates=10,
                 tau: float = 1e-2, batch_size: int = 512, update_frequency: int = 20,
                 critic_grad_norm_clip: int = 1, policy_update_frequency: int = 2,
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_shape (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(action_size=action_size, state_shape=state_shape, num_agents=num_agents)
        self.seed = random.seed(seed)
        self.n_seed = np.random.seed(seed)
        self.num_agents = num_agents
        self.num_learning_updates = num_learning_updates
        self.tau = tau

        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.critic_grad_norm_clip = critic_grad_norm_clip
        self.policy_update_frequency = policy_update_frequency

        self.policy = policy

        # critic local and target network (Q-Learning)
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
        # Replay memory
        self.memory = memory_factory()

    def set_mode(self, mode: str):
        if mode == 'train':
            self.online_actor.train()
            self.online_critic.train()
        elif mode == 'eval':
            self.online_actor.eval()
            self.online_critic.eval()
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

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
                    self.learn(experiences, kwargs['agent_number'])

    def step_episode(self, i_episode):
        # Reset the noise modules
        self.policy.step_episode(i_episode)

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

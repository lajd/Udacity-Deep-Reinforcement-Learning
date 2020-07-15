import torch.optim as optim
from tools.misc import *
from agents.models.components import noise as rm
from tools.rl_constants import Experience, ExperienceBatch
from agents.memory.memory import Memory

BUFFER_SIZE = int(1e6)  # replay buffer size
ACTOR_LR = 1e-3  # Actor network learning rate
CRITIC_LR = 1e-4  # Actor network learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGAgent:
    """Interacts with and learns from the environment."""

    online_critic = None
    online_actor = None

    target_critic = None
    target_actor = None

    critic_optimizer = None
    actor_optimizer = None

    memory = None
    policy = None

    def __init__(self, policy, state_size, action_size, num_agents, seed, fc1=400, fc2=300, update_times=10,
                 tau: float = 1e-2, batch_size: int = 512, num_learning_updates: int = 20,
                 critic_grad_norm_clip: int = 1, policy_update_frequency: int = 1):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.n_seed = np.random.seed(seed)
        self.num_agents = num_agents
        self.update_times = update_times
        self.t_step = 0
        self.tau = tau

        self.batch_size = batch_size
        self.num_learning_updates = num_learning_updates

        self.critic_grad_norm_clip = critic_grad_norm_clip
        self.policy_update_frequency = policy_update_frequency

        if MADDPGAgent.policy is None:
            MADDPGAgent.policy = policy
        # self.policy = policy

        # critic local and target network (Q-Learning)
        if MADDPGAgent.online_critic is None:
            MADDPGAgent.online_critic = Critic(state_size, action_size, fc1, fc2, seed).to(device)

        if MADDPGAgent.target_critic is None:
            MADDPGAgent.target_critic = Critic(state_size, action_size, fc1, fc2, seed).to(device)
            MADDPGAgent.target_critic.load_state_dict(MADDPGAgent.online_critic.state_dict())

        # actor local and target network (Policy gradient)
        if MADDPGAgent.online_actor is None:
            MADDPGAgent.online_actor = Actor(state_size, action_size, fc1, fc2, seed).to(device)
        if MADDPGAgent.target_actor is None:
            MADDPGAgent.target_actor = Actor(state_size, action_size, fc1, fc2, seed).to(device)
            MADDPGAgent.target_actor.load_state_dict(MADDPGAgent.online_actor.state_dict())

        # optimizer for critic and actor network
        if MADDPGAgent.critic_optimizer is None:
            MADDPGAgent.critic_optimizer = optim.Adam(MADDPGAgent.online_critic.parameters(), lr=CRITIC_LR, weight_decay=1.e-5)
        if MADDPGAgent.actor_optimizer is None:
            MADDPGAgent.actor_optimizer = optim.Adam(MADDPGAgent.online_actor.parameters(), lr=ACTOR_LR)

        # Replay memory
        if MADDPGAgent.memory is None:
            MADDPGAgent.memory = Memory(buffer_size=int(1e6), seed=0)

    def set_mode(self, mode: str):
        if mode == 'train':
            MADDPGAgent.online_actor.train()
            MADDPGAgent.online_critic.train()
        elif mode == 'eval':
            MADDPGAgent.online_actor.eval()
            MADDPGAgent.online_critic.eval()
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

    @staticmethod
    def preprocess_state(state):
        """ Perform any state preprocessing """
        return state

    def step(self, experience: Experience):
        # def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        # self.t_step += 1
        MADDPGAgent.memory.add(experience)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.num_learning_updates

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(MADDPGAgent.memory) > self.batch_size:
                for i in range(self.update_times):
                    experiences = MADDPGAgent.memory.sample(self.batch_size)
                    self.learn(experiences)

    def step_episode(self, i_episode):
        # Reset the noise modules
        self.policy.step_episode(i_episode)
        # [noise.reset_states() for noise in self.agent_noise]
        pass

    def get_action(self, state, training=True) -> np.ndarray:
        return self.policy.get_action(state, MADDPGAgent.online_actor, training=training)

    def get_random_action(self, *args) -> np.ndarray:
        """ Get a random action, used for warmup"""
        return self.policy.get_random_action()

    def learn(self, experience_batch: ExperienceBatch):

        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        experience_batch = experience_batch.to(device)
        critic_loss, critic_errors = MADDPGAgent.policy.compute_critic_errors(
            experience_batch,
            online_actor=MADDPGAgent.online_actor,
            online_critic=MADDPGAgent.online_critic,
            target_actor=MADDPGAgent.target_actor,
            target_critic=MADDPGAgent.target_critic,
        )

        MADDPGAgent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(MADDPGAgent.online_critic.parameters(), self.critic_grad_norm_clip)
        MADDPGAgent.critic_optimizer.step()

        if self.t_step % self.policy_update_frequency == 0:
            # Delay the policy update as in TD3
            actor_loss, actor_errors = MADDPGAgent.policy.compute_actor_errors(
                experience_batch,
                online_actor=MADDPGAgent.online_actor,
                online_critic=MADDPGAgent.online_critic,
                target_actor=MADDPGAgent.target_actor,
                target_critic=MADDPGAgent.target_critic,
            )

            MADDPGAgent.actor_optimizer.zero_grad()
            actor_loss.backward()
            MADDPGAgent.actor_optimizer.step()

            self.tau = min(5e-1, self.tau * 1.001)

            # Update target networks
            soft_update(MADDPGAgent.online_critic, MADDPGAgent.target_critic, self.tau)
            soft_update(MADDPGAgent.online_actor, MADDPGAgent.target_actor, self.tau)
            return critic_loss, critic_errors, actor_loss, actor_errors
        return critic_loss, critic_errors, None, None

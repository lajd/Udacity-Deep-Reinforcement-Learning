import torch.optim as optim
from tools.misc import *
from agents.models.components import noise as rm
from tools.rl_constants import Action, Experience, ExperienceBatch
from agents.memory.memory import Memory

BUFFER_SIZE = int(1e6)  # replay buffer size
ACTOR_LR = 1e-3  # Actor network learning rate
CRITIC_LR = 1e-4  # Actor network learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HomogeneousMADDPGAgent:
    """Interacts with and learns from the environment."""

    online_critic = None
    online_actor = None

    target_critic = None
    target_actor = None

    critic_optimizer = None
    actor_optimizer = None

    memory = None

    policy = None

    def __init__(self, policy, state_size, action_size, num_homogeneous_agents, seed, fc1=400, fc2=300, update_times=10,
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
        self.num_homogeneous_agents = num_homogeneous_agents
        self.update_times = update_times
        self.t_step = 0
        self.tau = tau

        self.batch_size = batch_size
        self.num_learning_updates = num_learning_updates

        self.critic_grad_norm_clip = critic_grad_norm_clip
        self.policy_update_frequency = policy_update_frequency

        if HomogeneousMADDPGAgent.policy is None:
            HomogeneousMADDPGAgent.policy = policy
        # self.policy = policy

        # critic local and target network (Q-Learning)
        if HomogeneousMADDPGAgent.online_critic is None:
            HomogeneousMADDPGAgent.online_critic = Critic(state_size, action_size, fc1, fc2, seed).to(device)

        if HomogeneousMADDPGAgent.target_critic is None:
            HomogeneousMADDPGAgent.target_critic = Critic(state_size, action_size, fc1, fc2, seed).to(device)
            HomogeneousMADDPGAgent.target_critic.load_state_dict(HomogeneousMADDPGAgent.online_critic.state_dict())

        # actor local and target network (Policy gradient)
        if HomogeneousMADDPGAgent.online_actor is None:
            HomogeneousMADDPGAgent.online_actor = Actor(state_size, action_size, fc1, fc2, seed).to(device)
        if HomogeneousMADDPGAgent.target_actor is None:
            HomogeneousMADDPGAgent.target_actor = Actor(state_size, action_size, fc1, fc2, seed).to(device)
            HomogeneousMADDPGAgent.target_actor.load_state_dict(HomogeneousMADDPGAgent.online_actor.state_dict())

        # optimizer for critic and actor network
        if HomogeneousMADDPGAgent.critic_optimizer is None:
            HomogeneousMADDPGAgent.critic_optimizer = optim.Adam(HomogeneousMADDPGAgent.online_critic.parameters(), lr=CRITIC_LR, weight_decay=1.e-5)
        if HomogeneousMADDPGAgent.actor_optimizer is None:
            HomogeneousMADDPGAgent.actor_optimizer = optim.Adam(HomogeneousMADDPGAgent.online_actor.parameters(), lr=ACTOR_LR)

        # Replay memory
        if HomogeneousMADDPGAgent.memory is None:
            HomogeneousMADDPGAgent.memory = Memory(buffer_size=int(1e6), seed=0)

    def set_mode(self, mode: str):
        if mode == 'train':
            HomogeneousMADDPGAgent.online_actor.train()
            HomogeneousMADDPGAgent.online_critic.train()
        elif mode == 'eval':
            HomogeneousMADDPGAgent.online_actor.eval()
            HomogeneousMADDPGAgent.online_critic.eval()
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
        HomogeneousMADDPGAgent.memory.add(experience)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.num_learning_updates

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(HomogeneousMADDPGAgent.memory) > self.batch_size:
                for i in range(self.update_times):
                    experiences = HomogeneousMADDPGAgent.memory.sample(self.batch_size)
                    self.learn(experiences)

    def step_episode(self, i_episode):
        # Reset the noise modules
        self.policy.step_episode(i_episode)
        # [noise.reset_states() for noise in self.agent_noise]
        pass

    def get_action(self, state, training=True):
        return self.policy.get_action(state, HomogeneousMADDPGAgent.online_actor, training=training)

    def get_random_action(self, *args):
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
        critic_loss, critic_errors = HomogeneousMADDPGAgent.policy.compute_critic_errors(
            experience_batch,
            online_actor=HomogeneousMADDPGAgent.online_actor,
            online_critic=HomogeneousMADDPGAgent.online_critic,
            target_actor=HomogeneousMADDPGAgent.target_actor,
            target_critic=HomogeneousMADDPGAgent.target_critic,
        )

        HomogeneousMADDPGAgent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(HomogeneousMADDPGAgent.online_critic.parameters(), self.critic_grad_norm_clip)
        HomogeneousMADDPGAgent.critic_optimizer.step()

        if self.t_step % self.policy_update_frequency == 0:
            # Delay the policy update as in TD3
            actor_loss, actor_errors = HomogeneousMADDPGAgent.policy.compute_actor_errors(
                experience_batch,
                online_actor=HomogeneousMADDPGAgent.online_actor,
                online_critic=HomogeneousMADDPGAgent.online_critic,
                target_actor=HomogeneousMADDPGAgent.target_actor,
                target_critic=HomogeneousMADDPGAgent.target_critic,
            )

            HomogeneousMADDPGAgent.actor_optimizer.zero_grad()
            actor_loss.backward()
            HomogeneousMADDPGAgent.actor_optimizer.step()

            self.tau = min(5e-1, self.tau * 1.001)

            # Update target networks
            soft_update(HomogeneousMADDPGAgent.online_critic, HomogeneousMADDPGAgent.target_critic, self.tau)
            soft_update(HomogeneousMADDPGAgent.online_actor, HomogeneousMADDPGAgent.target_actor, self.tau)
            return critic_loss, critic_errors, actor_loss, actor_errors
        return critic_loss, critic_errors, None, None

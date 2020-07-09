import torch.optim as optim
from tools.misc import *
from agents.models.components import noise as rm
from tools.rl_constants import Action, Experience, ExperienceBatch
from agents.memory.memory import Memory

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor
# TAU = 1e-3              # for soft update of target parameters
ACTOR_LR = 1e-3  # Actor network learning rate
CRITIC_LR = 1e-4  # Actor network learning rate
UPDATE_EVERY = 20  # how often to update the network (time step)
# UPDATE_TIMES = 5       # how many times to update in one go

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

    def __init__(self, state_size, action_size, num_homogeneous_agents, seed, fc1=400, fc2=300, update_times=10,
                 weight_decay=1.e-5, tau: float = 1e-3,):
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

        self.agent_noise = []
        for i in range(num_homogeneous_agents):
            self.agent_noise.append(rm.OrnsteinUhlenbeckProcess(size=(action_size,), std=LinearSchedule(0.4, 0, 2000)))

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

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.a_step = 0

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
        HomogeneousMADDPGAgent.memory.add(experience)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(HomogeneousMADDPGAgent.memory) > BATCH_SIZE:
                for i in range(self.update_times):
                    experiences = HomogeneousMADDPGAgent.memory.sample(BATCH_SIZE)
                    self.learn(experiences, GAMMA)

    def step_episode(self, i_episode):
        # Reset the noise modules
        [noise.reset_states() for noise in self.agent_noise]

    def get_action(self, state, training=True):
        self.t_step += 1
        assert state.shape[0] == self.num_homogeneous_agents, state.shape[0]

        epsilon = max((1500 - self.t_step) / 1500, .01)

        HomogeneousMADDPGAgent.online_actor.eval()
        with torch.no_grad():
            actions = HomogeneousMADDPGAgent.online_actor(state)
        HomogeneousMADDPGAgent.online_actor.train()

        if training:
            # return np.clip(actions.cpu().data.numpy()+np.random.uniform(-1,1,(2,2))*epsilon,-1,1) #adding noise to action space
            r = np.random.random()
            if r <= epsilon:
                action = np.random.uniform(-1, 1, (self.num_homogeneous_agents, self.action_size))
            else:
                action = np.clip(actions.cpu().data.numpy(), -1, 1)  # epsilon greedy policy
        else:
            action = actions.cpu().data.numpy()
        # raise RuntimeError(action.shape)
        action = Action(value=action)
        return action

    def learn(self, experience_batch: ExperienceBatch, gamma):

        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        experience_batch.to(device)
        batch_size = len(experience_batch)

        all_next_actions = HomogeneousMADDPGAgent.target_actor(experience_batch.joint_next_states.view(batch_size * 2, -1)).view(batch_size, -1)

        critic_target_input = torch.cat((experience_batch.joint_next_states, all_next_actions.view(batch_size * 2, -1)[1::2]), dim=1).to(
            device)
        with torch.no_grad():
            Q_target_next = HomogeneousMADDPGAgent.target_critic(critic_target_input, all_next_actions.view(batch_size * 2, -1)[::2])
        Q_targets = experience_batch.rewards + (gamma * Q_target_next * (1 - experience_batch.dones))

        critic_local_input = torch.cat((experience_batch.joint_states, experience_batch.joint_actions.view(batch_size * 2, -1)[1::2]), dim=1).to(device)
        Q_expected = HomogeneousMADDPGAgent.online_critic(critic_local_input, experience_batch.actions)

        # critic loss
        huber_loss = torch.nn.SmoothL1Loss()

        loss = huber_loss(Q_expected, Q_targets.detach())

        HomogeneousMADDPGAgent.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(HomogeneousMADDPGAgent.online_critic.parameters(), 1)
        HomogeneousMADDPGAgent.critic_optimizer.step()

        # actor loss

        action_pr_self = HomogeneousMADDPGAgent.online_actor(experience_batch.states)
        action_pr_other = HomogeneousMADDPGAgent.online_actor(experience_batch.joint_next_states.view(batch_size * 2, -1)[1::2]).detach()

        # critic_local_input2=torch.cat((all_state,torch.cat((action_pr_self,action_pr_other),dim=1)),dim=1)
        critic_local_input2 = torch.cat((experience_batch.joint_states, action_pr_other), dim=1)
        p_loss = -HomogeneousMADDPGAgent.online_critic(critic_local_input2, action_pr_self).mean()

        HomogeneousMADDPGAgent.actor_optimizer.zero_grad()
        p_loss.backward()

        HomogeneousMADDPGAgent.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.tau = min(5e-1, self.tau * 1.001)
        soft_update(HomogeneousMADDPGAgent.online_critic, HomogeneousMADDPGAgent.target_critic, self.tau)
        soft_update(HomogeneousMADDPGAgent.online_actor, HomogeneousMADDPGAgent.target_actor, self.tau)

    # def reset_random(self):
    #     for i in range(self.num_homogeneous_agents):
    #         self.agent_noise[i].reset_states()

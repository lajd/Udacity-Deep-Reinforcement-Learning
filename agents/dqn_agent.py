import numpy as np
import random
from collections import namedtuple, deque
from agents.base import Agent
from agents.policies.base import Policy
from tools.lr_scheduler import LRScheduler
from copy import deepcopy
import torch
import torch.nn.functional as F
from tools.memory import Memory
from tools.experience import Experience


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 model: torch.nn.Module,
                 policy: Policy,
                 lr_scheduler: LRScheduler,
                 optimizer: torch.optim.Optimizer,
                 replay_buffer_size: int = int(1e5),
                 batch_size: int = 32,
                 gamma: float = 0.95,
                 tau: float = 1e-3,
                 update_frequency: int = 5,
                 warmup_steps: int = 0,
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            layer_sizes (tuple[int]): Hidden layer sizes
            action_size (int): dimension of each action
        """
        super().__init__(
            state_size=state_size,
            action_size=state_size,
            policy=policy,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer
        )
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_frequency = update_frequency
        self.warmup_steps = warmup_steps

        self.qnetwork_local = model.to(device)
        self.qnetwork_target = deepcopy(model).to(device)

        # Replay memory
        self.memory = Memory(
            capacity=self.replay_buffer_size,
            state_shape=(1, state_size),
        )

        self.t_step = 0

    def get_agent_networks(self):
        return {
            "qnetwork_local": self.qnetwork_local
        }

    def load_agent_networks(self, agent_network_path_dict: dict):
        if 'qnetwork_local' not in agent_network_path_dict:
            raise ValueError("DQN agent expects network `qnetwork_local`")
        self.qnetwork_local.load_state_dict(torch.load(agent_network_path_dict['qnetwork_local']))
        self.set_mode('evaluate')

    def step_episode(self, episode: int):
        self.policy.step_episode(episode)
        self.lr_scheduler.step()
        return True

    def step(self, state: np.array, action: np.array, reward: np.array, next_state: np.array, done: np.array, **kwargs):
        """Step the agent in response to a change in environment"""
        # Save experience in replay memory

        # self.memory.add(state, action, reward, next_state, done)

        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)

        # # # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step > self.warmup_steps and self.t_step % self.update_frequency == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                states, actions, rewards, next_states, terminal, idxs, is_weights = self.memory.sample(self.batch_size)
                # experiences = self.memory.sample()
                experiences = (states, actions, rewards, next_states, terminal)
                td_error = self.learn(experiences, self.gamma)
                for i, (s, a, r, ns, t) in enumerate(zip(experiences)):
                    experience = Experience(state=s, action=a, reward=r, next_state=ns, done=t)
                    self.memory.add(experience.as_tuple(), td_error[i])
        else:
            # Default the priority to the reward
            self.memory.add(experience.as_tuple(), reward)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        action = self.policy.get_action(action_values=action_values)
        return action

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get the expected value of q for each action taken in `actions`
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Get the optimal (target) value of q(s', a')
        next_q_targets = gamma*self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma * next_q_targets * (1 - dones)

        # Calculate the TD error
        # td_error

        def huber_loss(loss):
            return 0.5 * loss ** 2 if abs(loss) < 1.0 else abs(loss) - 0.5

        td_errors = [huber_loss(q_targets - q_expected) for i in range(states.shape[0])]

        # Use weighted MSE loss
        weights = td_errors
        loss = torch.sum(weights * (q_expected - q_targets) ** 2)

        # # Back propagate
        # loss = F.mse_loss(q_expected, q_targets)
        #
        # def weighted_mse_loss(input, target, weight):
        #     return torch.sum(weight * (input - target) ** 2)

        self.optimizer.zero_grad()  # Zero gradients from previous step
        loss.backward()  # Compute the derivatives of the loss WRT the parameters (anything requiring gradients) using BP
        self.optimizer.step()  # Instruct the optimizer to take a step based on the gradients of the parameters

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return td_errors


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

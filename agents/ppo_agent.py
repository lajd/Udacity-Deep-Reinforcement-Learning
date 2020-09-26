from typing import Callable, List
from agents.models.ppo import PPO_Actor_Critic
from agents.memory.trajectories import Trajectories
import torch.nn as nn
from tools.rl_constants import Experience, Action, concatenate_action_attributes
import torch
from tools.parameter_scheduler import ParameterScheduler
from agents.base import Agent
from torch.nn import functional as F
from tools.misc import set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent(Agent):
    """Implements the PPO agent (https://openai.com/blog/openai-baselines-ppo/)
    with generalized advantage estimation (https://arxiv.org/pdf/1506.02438.pdf)
    """
    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int,
            actor_critic_factory: Callable[[], PPO_Actor_Critic],
            optimizer_factory: Callable[[torch.nn.Module.parameters], torch.optim.Optimizer],
            grad_clip: float = 1.,
            gamma: float = 0.99,
            batch_size: int = 1024,
            gae_factor: float = 0.95,
            epsilon: float = 0.2,
            beta_scheduler: ParameterScheduler = ParameterScheduler(
                initial=0.02,
                lambda_fn=lambda i: 0.02 * 0.995 ** i,
                final=1e-4
            ),
            std_scale_scheduler: ParameterScheduler = ParameterScheduler(
                initial=0.5,
                lambda_fn=lambda i: 0.5 * 0.995 ** i,
                final=0.2
            ),
            continuous_actions: bool = False,
            continuous_action_range_clip: tuple = (-1, 1),
            min_batches_for_training: int = 32,
            num_learning_updates: int = 4,
    ):
        """
        :param state_size: The state size of the agent
        :param action_size: The action size of the agent
        :param seed: Seed for reproducibility
        :param actor_critic_factory: Function returning the actor-critic model
        :param optimizer_factory: Function returning the optimizer for the actor-critic model
        :param grad_clip: Clip absolute value of the gradient above this value
        :param gamma: Discount factor
        :param batch_size: SGD minibatch size
        :param gae_factor: Factor used to down-weight rewards, presented as lambda in the GAE paper
        :param epsilon: Small constant parameter to clip the objective function by
        :param beta_scheduler: Scheduler for parameter beta, the coefficient for the entropy term
        :param std_scale_scheduler: Scheduler for the std of the normal distribution used to sample
            actions from in the policy network. Only used for continuous actions
        :param continuous_actions: Whether the action space is continuous or discrete
        :param continuous_action_range_clip: The range to clip continuous actions above. Only used for continuous actions
        :param min_batches_for_training: Minimum number of batches to accumulate before performing training
        :param num_learning_updates: Number of epochs to train for over before discarding samples
        """
        super().__init__(state_size, action_size)

        if seed is not None:
            set_seed(seed)
        self.online_actor_critic = actor_critic_factory().to(device)
        self.target_actor_critic = actor_critic_factory().to(device).eval()
        self.target_actor_critic.load_state_dict(self.online_actor_critic.state_dict())

        self.optimizer = optimizer_factory(self.online_actor_critic.parameters())
        self.current_trajectory_memory = Trajectories(seed)
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.batch_size = batch_size
        self.gae_factor = gae_factor

        self.beta_scheduler = beta_scheduler

        self.epsilon = epsilon
        self.beta = self.beta_scheduler.initial
        self.std_scale_scheduler = std_scale_scheduler
        self.std_scale = self.std_scale_scheduler.initial
        self.previous_std_scale = None

        self.continuous_actions = continuous_actions
        self.continuous_action_range_clip = continuous_action_range_clip

        self.min_batches_for_training = min_batches_for_training

        self.num_learning_updates = num_learning_updates

        self.warmup = False
        self.current_trajectory = []

    def set_mode(self, mode):
        if mode == 'train':
            self.online_actor_critic.train()

            # Check if we are switching to training from validation
            if self.std_scale == 0:
                self.std_scale = self.previous_std_scale
        elif mode == 'eval':
            self.online_actor_critic.eval()
            self.previous_std_scale = self.std_scale
            self.std_scale = 0
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

    def get_action(self, states, *args, **kwargs) -> Action:
        """Returns actions for given states as per target policy.
        :param states: States from environment
        :return: Action containing:
            - action (Tensor): predicted action
            - log_prob (Tensor): log probability of current action distribution
            - value (Tensor): estimate value function
        """
        # Use the target_actor_critic to get new actions
        states = states.to(device)
        self.target_actor_critic.eval()
        with torch.no_grad():
            actions, log_probs, _, values = self.target_actor_critic(state=states, scale=self.std_scale)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            actions = actions.cpu().data.numpy()
        self.target_actor_critic.train()
        if self.continuous_actions and self.continuous_action_range_clip:
            actions = actions.clip(self.continuous_action_range_clip[0], self.continuous_action_range_clip[1])
        return Action(value=actions, log_probs=log_probs, critic_values=values)

    def step(self, experience: Experience, *args, **kwargs):
        """ Add experience to current trajectory"""
        self.current_trajectory.append(experience)

    def compute_gae(self, next_value: List[torch.Tensor], rewards: List[torch.Tensor], masks: List[torch.Tensor], values: List[torch.Tensor]):
        """ Compute the generalized advantage estimate
        Adapted from https://github.com/higgsfield/RL-Adventure-2/blob/master/2.gae.ipynb
        and based off https://arxiv.org/pdf/1506.02438.pdf
        :param next_value: Value estimate of terminal state
        :param rewards: Trajectory rewards
        :param masks: Trajectory terminal states
        :param values: Trajectory value estimates
        :return: List of GAE returns
        """
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_factor * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def process_trajectory(self):
        """ Process the current trajectory and store in the trajectory buffer"""
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        joint_states = []
        joint_actions = []

        for i in range(len(self.current_trajectory)):
            experience = self.current_trajectory[i].to(device)

            action = concatenate_action_attributes(experience.action, attribute_name='value').to(device)
            critic_value = concatenate_action_attributes(experience.action, attribute_name='critic_values').to(device)
            log_prob = concatenate_action_attributes(experience.action, attribute_name='log_probs').to(device)
            terminal = (1-experience.done).to(device).view(-1, 1)
            reward = experience.reward.view(-1, 1)

            states.append(experience.state)
            log_probs.append(log_prob)
            actions.append(action)
            values.append(critic_value)
            rewards.append(reward)
            masks.append(terminal)

            joint_states.append(experience.joint_state.view(1, -1) if experience.joint_state is not None else None)
            joint_actions.append(experience.joint_action.view(1, -1) if experience.joint_action is not None else None)

        terminal_experience = self.current_trajectory[-1]
        next_value = self.get_action(
            terminal_experience.state,
            terminal_experience.joint_state.view(1, -1) if terminal_experience.joint_state is not None else None,
            terminal_experience.joint_action.view(1, -1) if terminal_experience.joint_action is not None else None,
            torch.from_numpy(terminal_experience.action.value).view(1, -1)
        ).critic_values

        returns = self.compute_gae(next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        joint_states = torch.cat(joint_states) if joint_states[0] is not None else joint_states
        joint_actions = torch.cat(joint_actions) if joint_actions[0] is not None else joint_actions

        advantage = returns - values

        processed_trajectory = list(zip(states, actions, log_probs, returns, advantage, joint_states, joint_actions))

        self.current_trajectory_memory.add(processed_trajectory)
        # reset trajectory
        self.current_trajectory = []

    def _learn(self, sampled_log_probs: torch.Tensor, sampled_states: torch.Tensor, sampled_actions: torch.Tensor, sampled_advantages: torch.Tensor, sampled_returns: torch.Tensor):
        """ Optimize the surrogate objective function over multiple epochs"""
        _, log_probs, entropy_loss, values = self.online_actor_critic(
            state=sampled_states, action=sampled_actions
        )

        sampled_log_probs = sampled_log_probs.view(-1, 1)
        log_probs = log_probs.view(-1, 1)

        # ratio for clipping
        ratio = (log_probs - sampled_log_probs.detach()).exp()
        # clipped function
        surrogate_1 = ratio * sampled_advantages
        surrogate_2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * sampled_advantages
        clipped_surrogate = -torch.min(surrogate_1, surrogate_2).mean(0)
        policy_loss = torch.mean(clipped_surrogate - self.beta * entropy_loss)
        value_loss = F.mse_loss(sampled_returns, values)

        # Update actor critic
        # Combine loss functions from actor/critic
        self.optimizer.zero_grad()
        (value_loss + policy_loss).backward()
        nn.utils.clip_grad_norm_(self.online_actor_critic.parameters(), self.grad_clip)
        self.optimizer.step()

    def step_episode(self, episode: int, *args, **kwargs):
        """ Perform end-of-episode updates """
        self.process_trajectory()
        if len(self.current_trajectory_memory) >= self.batch_size * self.min_batches_for_training:
            for _ in range(self.num_learning_updates):
                print('learning')
                for sampled_states, sampled_actions, sampled_log_probs, sampled_returns, sampled_advantages, _, _ in self.current_trajectory_memory.sample(self.batch_size):
                    self._learn(sampled_log_probs,  sampled_states, sampled_actions, sampled_advantages, sampled_returns)
            self.current_trajectory_memory.reset()
            # Hard update the target_actor_critic
            self.target_actor_critic.load_state_dict(self.online_actor_critic.state_dict())

        self.beta = self.beta_scheduler.get_param(episode)
        self.std_scale = self.std_scale_scheduler.get_param(episode)

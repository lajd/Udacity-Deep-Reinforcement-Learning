from typing import Callable
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
    """Interacts with and learns from the environment.

    Code partially adapted from https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
    """
    def __init__(
            self,
            state_size,
            action_size,
            random_seed,
            actor_critic_factory: lambda: PPO_Actor_Critic,
            optimizer_factory: Callable,
            grad_clip=1.,
            ppo_clip=0.2,
            gamma=0.99,
            batch_size=1024,
            gae_factor=0.95,
            epsilon=0.2,
            beta_scheduler=ParameterScheduler(initial=0.02,
                                              lambda_fn=lambda i: 0.02 * 0.995 ** i,
                                              final=1e-4),
            std_scale_scheduler=ParameterScheduler(initial=0.5,
                                              lambda_fn=lambda i: 0.5 * 0.995 ** i,
                                              final=0.2),
            continuous_actions: bool = False,
            continuous_action_range_clip: tuple = (-1, 1),
            min_batches_for_training=32,
            num_learning_updates=4,
            seed=None,
    ):
        super().__init__(state_size, action_size)

        if seed is not None:
            set_seed(seed)
        self.online_actor_critic = actor_critic_factory().to(device)
        self.target_actor_critic = actor_critic_factory().to(device).eval()
        self.target_actor_critic.load_state_dict(self.online_actor_critic.state_dict())

        self.optimizer = optimizer_factory(self.online_actor_critic.parameters())
        self.current_trajectory_memory = Trajectories(random_seed)
        self.grad_clip = grad_clip
        self.ppo_clip = ppo_clip
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

    def get_action(self, states, *args, **kwargs):
        """Returns actions for given states as per current policy.

        Returns
        ======
            action (Tensor): predicted action or inputed action
            log_prob (Tensor): log probability of current action distribution
            value (Tensor): estimate value function
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

    @staticmethod
    def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def process_trajectory(self):
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        joint_states = []
        joint_actions = []

        terminal_experience = self.current_trajectory[-1]

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

            if experience.joint_state is not None:
                joint_states.append(experience.joint_state.view(1, -1))
            if experience.joint_action is not None:
                joint_actions.append(experience.joint_action.view(1, -1))

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
        joint_states = torch.cat(joint_states) if len(joint_states) > 1 else None
        joint_actions = torch.cat(joint_actions) if len(joint_actions) > 1 else None

        advantage = returns - values
        new_results = []

        for i in range(len(states)):
            new_results.append(
                (
                    states[i],
                    actions[i],
                    log_probs[i],
                    returns[i],
                    advantage[i],
                    joint_states[i] if joint_states is not None else None,
                    joint_actions[i] if joint_actions is not None else None
                )
            )

        self.current_trajectory_memory.add(new_results)
        # reset trajectory
        self.current_trajectory = []

    def _learn(self, sampled_log_probs, sampled_states, sampled_actions, sampled_advantages, sampled_returns):

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

    def step_episode(self, episode, *args, **kwargs):
        self.process_trajectory()
        if len(self.current_trajectory_memory) >= self.batch_size * self.min_batches_for_training:
            for _ in range(self.num_learning_updates):
                for sampled_states, sampled_actions, sampled_log_probs, sampled_returns, sampled_advantages, _, _ in self.current_trajectory_memory.sample(self.batch_size):
                    self._learn(sampled_log_probs,  sampled_states, sampled_actions, sampled_advantages, sampled_returns)
            self.current_trajectory_memory.reset()
            # Hard update the target_actor_critic
            self.target_actor_critic.load_state_dict(self.online_actor_critic.state_dict())

        self.beta = self.beta_scheduler.get_param(episode)
        self.std_scale = self.std_scale_scheduler.get_param(episode)

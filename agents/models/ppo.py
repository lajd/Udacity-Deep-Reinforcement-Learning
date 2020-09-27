import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tools.misc import set_seed


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO_Actor_Critic(nn.Module):
    def __init__(self, actor_model, critic_model, action_size, continuous_actions: bool, initial_std=0.2, continuous_action_range_clip: Optional[tuple] = (-1, 1), seed=None):
        super(PPO_Actor_Critic, self).__init__()
        if seed is not None:
            set_seed(seed)
        self.actor = actor_model
        self.critic = critic_model
        self.action_size = action_size
        self.continuous_actions = continuous_actions
        self.std = nn.Parameter(torch.ones(1, action_size) * initial_std)
        self.continuous_action_range_clip = continuous_action_range_clip

    def step_episode(self):
        pass

    def forward(self, state, action=None, scale=1, min_std=0.05, *args, **kargs):
        assert min_std >= 0 and scale >= 0
        if self.continuous_actions:
            action_mean = self.actor(state)
            std = F.hardtanh(self.std, min_val=min_std, max_val=scale)
            dist = torch.distributions.Normal(action_mean, std)
        else:
            action_probs = self.actor(state)
            dist = torch.distributions.Categorical(probs=action_probs)

        if action is None:
            action = dist.sample()

        log_probs = torch.sum(dist.log_prob(action), dim=1, keepdim=True)
        dist_entropy = dist.entropy().mean()

        critic_value = self.critic(state)
        # critic_value = self.critic(state, action)
        if self.continuous_actions and self.continuous_action_range_clip:
            action = action.clamp(self.continuous_action_range_clip[0], self.continuous_action_range_clip[1])
        return action, log_probs, dist_entropy, critic_value


class MAPPO_Actor_Critic(nn.Module):
    def __init__(self, actor_model, critic_model, action_size, continuous_actions: bool, initial_std=0.2, continuous_action_range_clip: Optional[tuple] = (-1, 1), seed=None):
        super(MAPPO_Actor_Critic, self).__init__()
        if seed is not None:
            set_seed(seed)
        self.actor = actor_model
        self.critic = critic_model
        self.action_size = action_size
        self.continuous_actions = continuous_actions
        self.std = nn.Parameter(torch.ones(1, action_size) * initial_std)
        self.continuous_action_range_clip = continuous_action_range_clip

    def step_episode(self):
        pass

    def forward(self, agent_state: torch.FloatTensor, other_agent_states: torch.FloatTensor,
                other_agent_actions: Optional[torch.FloatTensor] = None, action: Optional[torch.FloatTensor] = None,  min_std=0.05, scale=1,):
        assert min_std > 0 and scale >= 0, (min_std, scale)

        if self.continuous_actions:
            action_mean = self.actor(agent_state)
            std = F.hardtanh(self.std, min_val=min_std, max_val=scale)
            dist = torch.distributions.Normal(action_mean, std)
        else:
            action_probs = self.actor(agent_state)
            dist = torch.distributions.Categorical(probs=action_probs)

        if action is None:
            action = dist.sample().to(device)

        if action.ndim > 1:
            action = action.squeeze().to(device)
        if other_agent_actions is None:
            if self.continuous_actions:
                other_agent_action_mean = self.actor(agent_state)
                std = F.hardtanh(self.std, min_val=min_std, max_val=scale)
                other_agent_dist = torch.distributions.Normal(other_agent_action_mean, std)
            else:
                other_action_probs = self.actor(other_agent_states)
                other_agent_dist = torch.distributions.Categorical(probs=other_action_probs)
            other_agent_actions = other_agent_dist.sample().to(device)

        critic_value = self.critic(agent_state, other_agent_states, other_agent_actions, action)

        log_probs = dist.log_prob(action)

        dist_entropy = dist.entropy().mean()
        return action, log_probs, dist_entropy, critic_value

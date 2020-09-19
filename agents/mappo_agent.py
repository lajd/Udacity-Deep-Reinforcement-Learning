import torch
import torch.nn as nn
from typing import Callable
from agents.models.ppo import PPO_Actor_Critic
from agents.ppo_agent import PPOAgent
from typing import Optional
from tools.parameter_scheduler import ParameterScheduler
from tools.rl_constants import Action
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MAPPOAgent(PPOAgent):
    """Interacts with and learns from the environment."""
    def __init__(
            self,
            agent_id,
            state_size,
            action_size,
            random_seed,
            actor_critic_factory: lambda: PPO_Actor_Critic,
            optimizer_factory: Callable,
            map_agent_to_state_slice: dict,
            map_agent_to_action_slice: dict,
            grad_clip=1.,
            ppo_clip=0.2,
            gamma=0.99,
            batch_size=1024,
            gae_factor=0.95,
            epsilon=0.2,
            beta_scheduler=ParameterScheduler(initial=0.015,
                                              lambda_fn=lambda i: 0.015 * 0.998 ** i,
                                              final=1e-6),
            std_scale_scheduler=ParameterScheduler(initial=0.8,
                                              lambda_fn=lambda i: 0.8 * 0.999 ** i,
                                              final=0.2),
            continuous_actions: bool = False,
            continuous_action_range_clip: tuple = (-1, 1),
            min_batches_for_training=16,
            num_learning_updates=10,
            seed=None,
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super().__init__(
            state_size,
            action_size,
            random_seed,
            actor_critic_factory,
            optimizer_factory,
            grad_clip,
            ppo_clip,
            gamma,
            batch_size,
            gae_factor,
            epsilon,
            beta_scheduler,
            std_scale_scheduler,
            continuous_actions,
            continuous_action_range_clip,
            min_batches_for_training,
            num_learning_updates,
            seed
        )

        self.agent_id = agent_id
        self.map_agent_to_state_slice = map_agent_to_state_slice
        self.map_agent_to_action_slice = map_agent_to_action_slice

    def get_action(self, agent_state: torch.FloatTensor, joint_state: torch.FloatTensor, joint_action: Optional[torch.FloatTensor]=None, action: Optional[torch.FloatTensor]=None, *args, **kwargs) -> Action:
        """Returns actions for given states as per current policy.

        Returns
        ======
            action (Tensor): predicted action or inputed action
            log_prob (Tensor): log probability of current action distribution
            value (Tensor): estimate value function
        """
        other_agent_states = self.get_other_agent_atributes(joint_state, self.map_agent_to_state_slice, flatten=False)
        other_agent_actions = self.get_other_agent_atributes(joint_action, self.map_agent_to_action_slice, flatten=False) if joint_action is not None else None

        self.target_actor_critic.eval()
        with torch.no_grad():
            actions, log_probs, _, values = self.target_actor_critic(
                agent_state=agent_state, other_agent_states=other_agent_states,
                other_agent_actions=other_agent_actions, action=action, scale=self.std_scale
            )
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            actions = actions.cpu().data.numpy()
        self.target_actor_critic.train()
        if self.continuous_actions and self.continuous_action_range_clip:
            actions = actions.clip(self.continuous_action_range_clip[0], self.continuous_action_range_clip[1])

        return Action(value=actions, log_probs=log_probs, critic_values=values)

    def get_other_agent_atributes(self, x: torch.Tensor, agent_slicing_dict: dict, apply_fn_map: dict = None, flatten: bool = True):
        output = []
        for k, f in agent_slicing_dict.items():
            if k != self.agent_id:
                if apply_fn_map:
                    output.append(apply_fn_map[k](f(x)).reshape(1, -1))
                else:
                    output.append(f(x).reshape(1, -1))

        if flatten:
            output = torch.cat(output, dim=1)
        else:
            output = torch.cat(output, dim=0)
        return output

    def get_agent_attributes(self, x: torch.Tensor, agent_slicing_dict: dict):
        return agent_slicing_dict[self.agent_id](x)

    def _learn(self, sampled_log_probs, sampled_joint_states, sampled_joint_actions, sampled_states, sampled_actions, sampled_advantages, sampled_returns):
        other_agent_states = self.get_other_agent_atributes(sampled_joint_states, self.map_agent_to_state_slice, flatten=False)
        other_agent_actions = self.get_other_agent_atributes(sampled_joint_actions, self.map_agent_to_action_slice, flatten=False)

        bsize = len(sampled_states)

        _, log_probs, entropy_loss, values = self.online_actor_critic(
            agent_state=sampled_states, other_agent_states=other_agent_states,
            other_agent_actions=other_agent_actions, action=sampled_actions
        )
        sampled_log_probs = sampled_log_probs.view(bsize, -1)
        log_probs = log_probs.view(bsize, -1)

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

    def step_episode(self, episode, *args):
        self.process_trajectory()
        if len(self.current_trajectory_memory) >= self.batch_size * self.min_batches_for_training:
            for _ in range(self.num_learning_updates):
                for sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages, joint_states, joint_actions in self.current_trajectory_memory.sample(self.batch_size):
                    self._learn(sampled_log_probs_old, joint_states, joint_actions, sampled_states, sampled_actions, sampled_advantages, sampled_returns)
            self.current_trajectory_memory.reset()
            # Hard update the target_actor_critic
            self.target_actor_critic.load_state_dict(self.online_actor_critic.state_dict())

        self.beta = self.beta_scheduler.get_param(episode)
        self.std_scale = self.std_scale_scheduler.get_param(episode)

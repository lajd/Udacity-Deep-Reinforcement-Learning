import torch
import torch.nn as nn
from typing import Callable
from agents.models.ppo import PPO_Actor_Critic
from agents.ppo_agent import PPOAgent
from typing import Optional, Dict
from tools.parameter_scheduler import ParameterScheduler
from tools.rl_constants import Action
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MAPPOAgent(PPOAgent):
    """Implements the multi-agent PPO agent"""
    def __init__(
            self,
            agent_id: str,
            state_size: int,
            action_size: int,
            actor_critic_factory: Callable[[], PPO_Actor_Critic],
            optimizer_factory: Callable[[torch.nn.Module.parameters], torch.optim.Optimizer],
            map_agent_to_state_slice: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
            map_agent_to_action_slice: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
            grad_clip: float = 1.,
            gamma: float = 0.99,
            batch_size: int = 1024,
            gae_factor: float = 0.95,
            epsilon: float = 0.2,
            beta_scheduler: ParameterScheduler = ParameterScheduler(
                initial=0.015,
                lambda_fn=lambda i: 0.015 * 0.998 ** i,
                final=1e-6
            ),
            std_scale_scheduler: ParameterScheduler = ParameterScheduler(
                initial=0.8,
                lambda_fn=lambda i: 0.8 * 0.999 ** i,
                final=0.2
            ),
            continuous_actions: bool = False,
            continuous_action_range_clip: tuple = (-1, 1),
            min_batches_for_training: int = 16,
            num_learning_updates: int = 10,
            seed: Optional[int] = None,
    ):
        """
        :param agent_id: The identifier for the agent, used to identify other agents' states/actions
        :param state_size: The state size of the agent
        :param action_size: The action size of the agent
        :param seed: Seed for reproducibility
        :param actor_critic_factory: Function returning the actor-critic model
        :param optimizer_factory: Function returning the optimizer for the actor-critic model
        :param map_agent_to_state_slice: Dictionary mapping the agent_id to a function which slices the joint_state
         tensor such that it extracts the agents state
        :param map_agent_to_action_slice: Dictionary mapping the agent_id to a function which slices the joint_action
         tensor such that it extracts the agents action
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
        super().__init__(
            state_size,
            action_size,
            seed,
            actor_critic_factory,
            optimizer_factory,
            grad_clip,
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
        )

        self.agent_id = agent_id
        self.map_agent_to_state_slice = map_agent_to_state_slice
        self.map_agent_to_action_slice = map_agent_to_action_slice

    def get_action(self, agent_state: torch.FloatTensor, joint_state: torch.FloatTensor, joint_action: Optional[torch.FloatTensor]=None, action: Optional[torch.FloatTensor]=None, *args, **kwargs) -> Action:
        """Returns actions for given states as per target policy.
        :param agent_state: States for this agent
        :param joint_state: States for all agents
        :param joint_action: Actions for all agents
        :param action: Action for this agent

        :return: Action containing:
            - action (Tensor): predicted action
            - log_prob (Tensor): log probability of current action distribution
            - value (Tensor): estimate value function
        """
        other_agent_states = self.get_other_agent_attributes(joint_state, self.map_agent_to_state_slice, flatten=False)
        other_agent_actions = self.get_other_agent_attributes(joint_action, self.map_agent_to_action_slice, flatten=False) if joint_action is not None else None

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

    def get_other_agent_attributes(self, x: torch.Tensor, agent_slicing_dict: Dict[str, Callable[[torch.Tensor], torch.Tensor]], apply_fn_map: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = None, flatten: bool = True):
        """ Get the attributes for all other agents
        :param x: Tensor containing states or actions
        :param agent_slicing_dict: Dictionary mapping the agent_id to a function which slices the tensor to extract
            agent attributes
        :param apply_fn_map: Mapping from agent_id to a function for pre-processing a tensor
        :param flatten: Whether to flatten the joint attributes into a 1d vector
        :return: torch.Tensor of other agent attributes
        """
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
        """ Get the agent's attributes
        :param x: Tensor containing states or actions
        :param agent_slicing_dict: Dictionary mapping the agent_id to a function which slices the tensor to extract
            agent attributes
        """
        return agent_slicing_dict[self.agent_id](x)

    def _learn(self, sampled_log_probs: torch.Tensor, sampled_joint_states: torch.Tensor, sampled_joint_actions: torch.Tensor, sampled_states: torch.Tensor, sampled_actions: torch.Tensor, sampled_advantages: torch.Tensor, sampled_returns: torch.Tensor):
        other_agent_states = self.get_other_agent_attributes(sampled_joint_states, self.map_agent_to_state_slice, flatten=False)
        other_agent_actions = self.get_other_agent_attributes(sampled_joint_actions, self.map_agent_to_action_slice, flatten=False)

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

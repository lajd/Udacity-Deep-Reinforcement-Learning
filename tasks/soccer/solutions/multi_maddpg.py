import os
from os.path import join
import pickle
import numpy as np
import torch.optim as optim
import torch
from tools.rl_constants import Experience, Brain, BrainSet
from tasks.soccer.solutions.utils import STRIKER_STATE_SIZE, GOALIE_STATE_SIZE, NUM_STRIKER_AGENTS,\
    NUM_GOALIE_AGENTS, get_simulator, GOALIE_BRAIN_NAME, STRIKER_BRAIN_NAME, GOALIE_ACTION_SIZE, STRIKER_ACTION_SIZE, \
    GOALIE_ACTION_DISCRETE_RANGE, STRIKER_ACTION_DISCRETE_RANGE
from agents.maddpg_agent import MADDPGAgent
from tasks.tennis.solutions.maddpg import SOLUTIONS_CHECKPOINT_DIR
from agents.policies.maddpg_policy import MADDPGPolicy
from tools.misc import LinearSchedule
from agents.models.components import noise as rm
from tools.misc import *
from agents.memory.memory import Memory

SAVE_TAG = 'homogeneous_maddpg_baseline'
ACTOR_CHECKPOINT_FN = lambda brain_name: join(SOLUTIONS_CHECKPOINT_DIR, f'{brain_name}_{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT_FN = lambda brain_name: join(SOLUTIONS_CHECKPOINT_DIR, f'{brain_name}_{SAVE_TAG}_critic_checkpoint.pth')
TRAINING_SCORES_SAVE_PATH_FN = lambda brain_name: join(SOLUTIONS_CHECKPOINT_DIR, f'{brain_name}_{SAVE_TAG}_training_scores.pkl')


NUM_EPISODES = 1000
MAX_T = 1000
SOLVE_SCORE = 2
WARMUP_STEPS = 1#5000
BUFFER_SIZE = int(1e6)  # replay buffer size
ACTOR_LR = 1e-3  # Actor network learning rate
CRITIC_LR = 1e-4  # Actor network learning rate
SEED = 0


def step_agents_fn(brain_set: BrainSet, next_brain_environment: dict, t: int):
    for brain_name, brain_environment in next_brain_environment.items():
        for i in range(brain_set[brain_name].num_agents):
            joint_state = torch.cat((brain_environment['states'][i], brain_environment['states'][1 - i]))
            joint_action = np.concatenate((brain_environment['actions'][i], brain_environment['actions'][1 - i]))
            join_next_state = torch.cat((brain_environment['next_states'][i], brain_environment['next_states'][1 - i]))

            brain_agent_experience = Experience(
                state=brain_environment['states'][i],
                action=brain_environment['actions'][i],
                reward=brain_environment['rewards'][i],
                next_state=brain_environment['next_states'][i],
                done=brain_environment['dones'][i],
                t_step=t,
                joint_state=joint_state,
                joint_action=torch.from_numpy(joint_action),
                joint_next_state=join_next_state
            )
            brain_set[brain_name].agent.step(brain_agent_experience)


if __name__ == '__main__':
    simulator = get_simulator()

    goalie_maddpg_agent = MADDPGAgent(
        policy=MADDPGPolicy(
            noise_factory=lambda: rm.OrnsteinUhlenbeckProcess(size=(GOALIE_ACTION_SIZE,), std=LinearSchedule(0.4, 0, 2000)),
            action_dim=GOALIE_ACTION_SIZE,
            num_agents=NUM_GOALIE_AGENTS,
            continuous_actions=False,
            discrete_action_range=GOALIE_ACTION_DISCRETE_RANGE
        ),
        state_shape=GOALIE_STATE_SIZE,
        action_size=GOALIE_ACTION_SIZE,
        num_agents=NUM_GOALIE_AGENTS,
        critic_factory=lambda: Critic(GOALIE_STATE_SIZE, GOALIE_STATE_SIZE, 400, 300, seed=SEED),
        actor_factory=lambda: Actor(GOALIE_STATE_SIZE, GOALIE_ACTION_SIZE, 400, 300, seed=SEED, with_argmax=True),
        critic_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=CRITIC_LR, weight_decay=1.e-5),
        actor_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=ACTOR_LR),
        memory_factory=lambda: Memory(buffer_size=BUFFER_SIZE, seed=SEED),
        seed=0,
    )

    goalie_brain = Brain(
        brain_name=GOALIE_BRAIN_NAME,
        action_size=GOALIE_ACTION_SIZE,
        state_shape=GOALIE_STATE_SIZE,
        observation_type='vector',
        agent=goalie_maddpg_agent,
        num_agents=NUM_GOALIE_AGENTS,
    )

    striker_maddpg_agent = MADDPGAgent(
        policy=MADDPGPolicy(
            noise_factory=lambda: rm.OrnsteinUhlenbeckProcess(size=(STRIKER_ACTION_SIZE,), std=LinearSchedule(0.4, 0, 2000)),
            action_dim=STRIKER_ACTION_SIZE,
            num_agents=NUM_STRIKER_AGENTS,
            continuous_actions=False,
            discrete_action_range=STRIKER_ACTION_DISCRETE_RANGE
        ),
        state_shape=STRIKER_STATE_SIZE,
        action_size=STRIKER_ACTION_SIZE,
        num_agents=NUM_STRIKER_AGENTS,
        critic_factory=lambda: Critic(STRIKER_STATE_SIZE, STRIKER_ACTION_SIZE, 400, 300, seed=SEED),
        actor_factory=lambda: Actor(STRIKER_STATE_SIZE, STRIKER_ACTION_SIZE, 400, 300, seed=SEED, with_argmax=True),
        critic_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=CRITIC_LR, weight_decay=1.e-5),
        actor_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=ACTOR_LR),
        memory_factory=lambda: Memory(buffer_size=BUFFER_SIZE, seed=SEED),
        seed=0,
    )

    striker_brain = Brain(
        brain_name=STRIKER_BRAIN_NAME,
        action_size=STRIKER_ACTION_SIZE,
        state_shape=STRIKER_STATE_SIZE,
        observation_type='vector',
        agent=striker_maddpg_agent,
        num_agents=NUM_STRIKER_AGENTS
    )

    brain_set = BrainSet(brains=[goalie_brain, striker_brain])

    simulator.warmup(brain_set, step_agents_fn=step_agents_fn, n_episodes=int(WARMUP_STEPS / MAX_T), max_t=MAX_T)

    brain_set, training_scores, i_episode, training_time = simulator.train(
        brain_set,
        n_episodes=NUM_EPISODES,
        max_t=MAX_T,
        solved_score=SOLVE_SCORE,
        step_agents_fn=step_agents_fn,
        reward_accumulation_fn=lambda rewards: np.max(rewards),
    )

    if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
        for brain_name, brain in brain_set:
            trained_agent = brain.agent
            torch.save(trained_agent.online_actor.state_dict(), ACTOR_CHECKPOINT_FN(brain_name))
            torch.save(trained_agent.online_critic.state_dict(), CRITIC_CHECKPOINT_FN(brain_name))
            with open(TRAINING_SCORES_SAVE_PATH_FN(brain_name), 'wb') as f:
                pickle.dump(training_scores, f)

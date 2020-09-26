import torch
from os.path import join
import pickle
import numpy as np
import torch.optim as optim
from tools.rl_constants import Brain, BrainSet
from tasks.soccer.solutions.utils import STRIKER_STATE_SIZE, GOALIE_STATE_SIZE, NUM_STRIKER_AGENTS, \
    get_simulator, NUM_GOALIE_AGENTS, GOALIE_ACTION_SIZE, STRIKER_ACTION_SIZE, GOALIE_ACTION_DISCRETE_RANGE,\
    STRIKER_ACTION_DISCRETE_RANGE, STRIKER_BRAIN_NAME, GOALIE_BRAIN_NAME
from tasks.tennis.solutions.maddpg import SOLUTIONS_CHECKPOINT_DIR
from tools.parameter_scheduler import ParameterScheduler
from agents.memory.prioritized_memory import PrioritizedMemory

from tools.rl_constants import RandomBrainAction
from agents.maddpg_agent import MADDPGAgent, DummyMADDPGAgent
from agents.policies.independent_maddpg_policy import IndependentMADDPGPolicy
from agents.models.components.mlp import MLP
from agents.models.components.critics import MACritic
from agents.models.td3 import TD3Critic
from agents.models.components.misc import SoftmaxSelection
from simulation.utils import multi_agent_step_agents_fn

SAVE_TAG = 'independent_maddpg_baseline'
ACTOR_CHECKPOINT_FN = lambda brain_name, agent_num: join(SOLUTIONS_CHECKPOINT_DIR, f'{brain_name}_agent_{agent_num}_{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT_FN = lambda brain_name, agent_num: join(SOLUTIONS_CHECKPOINT_DIR, f'{brain_name}_agent_{agent_num}_{SAVE_TAG}_critic_checkpoint.pth')
TRAINING_SCORES_FIGURE_SAVE_PATH = join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.png')
TRAINING_SCORES_SAVE_PATH = join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')

NUM_EPISODES = 1000
MAX_T = 1000
SOLVE_SCORE = 100
WARMUP_STEPS = 5000
BUFFER_SIZE = int(1e6)  # replay buffer size
ACTOR_LR = 1e-3  # Actor network learning rate
CRITIC_LR = 1e-4  # Actor network learning rate
SEED = 0
BATCH_SIZE = 128
TAU = 1e-2
BATCHNORM = False
TOTAL_NUM_AGENTS = 4
DROPOUT = None

TD3: bool = False

"""
RED: Agent 0
Blue: Agent 1
"""


if __name__ == '__main__':

    memory_fn = lambda: PrioritizedMemory(
        capacity=BUFFER_SIZE,
        seed=SEED,
        state_shape=(1,),
        alpha_scheduler=ParameterScheduler(initial=0.6, lambda_fn=lambda i: 0.6 - 0.6 * i / NUM_EPISODES, final=0.),
        beta_scheduler=ParameterScheduler(initial=0.4, final=1,
                                          lambda_fn=lambda i: 0.4 + 0.6 * i / NUM_EPISODES),  # Anneal beta linearly
        continuous_actions=False,
        min_priority=1e-9
    )

    simulator = get_simulator()

    critic = MACritic(
        state_featurizer=MLP(
            layer_sizes=(336*4 + 3, 256, 128),
            with_batchnorm=BATCHNORM,
            dropout=DROPOUT,
            seed=SEED
        ),
        output_module=MLP(
            layer_sizes=(128 + 1, 128, 1),
            with_batchnorm=BATCHNORM,
            dropout=DROPOUT,
            seed=SEED,
        ),
    )
    if TD3:
        critic_model = TD3Critic(
            critic_model_factory=lambda: critic
        )
    else:
        critic_model=critic

    goalie_agents = []
    for agent_num in range(NUM_GOALIE_AGENTS):
        key = 'GoalieBrain_{}'.format(agent_num)
        if agent_num == 1:
            goalie_agent = DummyMADDPGAgent(
                GOALIE_STATE_SIZE,
                len(range(*GOALIE_ACTION_DISCRETE_RANGE)),
                SEED,
                map_agent_to_state_slice={
                    "GoalieBrain_0": lambda t: t[:, 0:336],
                    "GoalieBrain_1": lambda t: t[:, 336:672],
                    "StrikerBrain_0": lambda t: t[:, 672:1008],
                    "StrikerBrain_1": lambda t: t[:, 1008:]
                },
                map_agent_to_action_slice={
                    "GoalieBrain_0": lambda t: t[:, 0:1],
                    "GoalieBrain_1": lambda t: t[:, 1:2],
                    "StrikerBrain_0": lambda t: t[:, 2:3],
                    "StrikerBrain_1": lambda t: t[:, 3:4]
                },
            )
        else:
            goalie_agent = MADDPGAgent(
                None, GOALIE_STATE_SIZE, GOALIE_ACTION_SIZE,
                critic_factory=lambda: critic_model,
                actor_factory=lambda: MLP(layer_sizes=(GOALIE_STATE_SIZE, 256, 128, len(range(*GOALIE_ACTION_DISCRETE_RANGE))), seed=SEED, output_function=SoftmaxSelection()),
                critic_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=CRITIC_LR, weight_decay=1.e-5),
                actor_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=ACTOR_LR),
                memory_factory=memory_fn,
                seed=SEED,
                batch_size=BATCH_SIZE,
                tau=TAU,
                num_learning_updates=10,
                update_frequency=20,
                policy_update_frequency=2,
                action_size=GOALIE_ACTION_SIZE,
            )

        goalie_agents.append(goalie_agent)

    striker_agents = []
    for agent_num in range(NUM_STRIKER_AGENTS):
        key = 'StrikerBrain_{}'.format(agent_num)
        if agent_num == 1:
            striker_agent = DummyMADDPGAgent(
                STRIKER_STATE_SIZE,
                len(range(*STRIKER_ACTION_DISCRETE_RANGE)),
                SEED,
                map_agent_to_state_slice={
                    "GoalieBrain_0": lambda t: t[:, 0:336],
                    "GoalieBrain_1": lambda t: t[:, 336:672],
                    "StrikerBrain_0": lambda t: t[:, 672:1008],
                    "StrikerBrain_1": lambda t: t[:, 1008:]
                },
                map_agent_to_action_slice={
                    "GoalieBrain_0": lambda t: t[:, 0:1],
                    "GoalieBrain_1": lambda t: t[:, 1:2],
                    "StrikerBrain_0": lambda t: t[:, 2:3],
                    "StrikerBrain_1": lambda t: t[:, 3:4]
                },
            )
        else:
            striker_agent = MADDPGAgent(
                None, STRIKER_STATE_SIZE, STRIKER_ACTION_SIZE,
                critic_factory=lambda: critic_model,
                actor_factory=lambda: MLP(layer_sizes=(GOALIE_STATE_SIZE, 256, 128, len(range(*STRIKER_ACTION_DISCRETE_RANGE))), seed=SEED, output_function=SoftmaxSelection()),
                critic_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=CRITIC_LR, weight_decay=1.e-5),
                actor_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=ACTOR_LR),
                memory_factory=memory_fn,
                seed=SEED,
                batch_size=BATCH_SIZE,
                tau=TAU,
                num_learning_updates=10,
                update_frequency=20,
                policy_update_frequency=2,
                action_size=STRIKER_ACTION_SIZE
            )

        striker_agents.append(striker_agent)

    goalie_brain = Brain(
        brain_name=GOALIE_BRAIN_NAME,
        action_size=GOALIE_ACTION_SIZE,
        state_shape=GOALIE_STATE_SIZE,
        observation_type='vector',
        agents=goalie_agents,
    )

    striker_brain = Brain(
        brain_name=STRIKER_BRAIN_NAME,
        action_size=STRIKER_ACTION_SIZE,
        state_shape=STRIKER_STATE_SIZE,
        observation_type='vector',
        agents=striker_agents,
    )

    brain_set = BrainSet(brains=[goalie_brain, striker_brain])

    for brain_name, brain in brain_set:
        for agent_num, agent in enumerate(brain.agents):
            agent_id = "{}_{}".format(brain_name, agent_num)
            if brain_name == 'GoalieBrain':
                action_size = GOALIE_ACTION_SIZE
                action_range = GOALIE_ACTION_DISCRETE_RANGE
            elif brain_name == 'StrikerBrain':
                action_size = STRIKER_ACTION_SIZE
                action_range = STRIKER_ACTION_DISCRETE_RANGE
            else:
                raise ValueError('fuck')

            agent.policy = IndependentMADDPGPolicy(
                brain_set=brain_set,
                agent_id=agent_id,
                action_dim=action_size,
                epsilon_scheduler=ParameterScheduler(initial=1, lambda_fn=lambda i: 0.98**i, final=0.01),
                random_brain_action_factory=lambda: RandomBrainAction(
                    1,
                    1,
                    continuous_actions=False,
                    continuous_action_range=None,
                    discrete_action_range=action_range
                ),
                map_agent_to_state_slice={
                    "GoalieBrain_0": lambda t: t[:, 0:336],
                    "GoalieBrain_1": lambda t: t[:, 336:672],
                    "StrikerBrain_0": lambda t: t[:, 672:1008],
                    "StrikerBrain_1": lambda t: t[:, 1008:]
                },
                map_agent_to_action_slice={
                    "GoalieBrain_0": lambda t: t[:, 0:1],
                    "GoalieBrain_1": lambda t: t[:, 1:2],
                    "StrikerBrain_0": lambda t: t[:, 2:3],
                    "StrikerBrain_1": lambda t: t[:, 3:4]
                },
                matd3=TD3,
                continuous_actions=False,
                continuous_actions_clip_range=None
            )

    simulator.warmup(
        brain_set,
        step_agents_fn=multi_agent_step_agents_fn,
        n_episodes=int(WARMUP_STEPS / MAX_T),
        max_t=MAX_T,
    )

    def episode_reward_fn(brain_episode_scores):
        team_scores = np.zeros(2)
        for brain_name_ in brain_episode_scores:
            agent_scores = brain_episode_scores[brain_name_]
            team_scores += agent_scores

        return round(team_scores[0])

    def brain_episode_rewards(rewards):
        return np.array(rewards)


    brain_set, training_scores, i_episode, training_time = simulator.train(
        brain_set,
        n_episodes=NUM_EPISODES,
        max_t=MAX_T,
        solved_score=SOLVE_SCORE,
        step_agents_fn=multi_agent_step_agents_fn,
        brain_reward_accumulation_fn=lambda rewards: brain_episode_rewards(rewards),  # Sum striker/goalie rewards
        episode_reward_accumulation_fn=lambda brain_episode_scores: episode_reward_fn(brain_episode_scores),  # Take the max score between the two teams
    )

    if training_scores.get_mean_sliding_scores() >= SOLVE_SCORE:
        for brain_name, brain in brain_set:
            for agent_num, agent in enumerate(brain.agents):
                if agent_num == 0:
                    # Only AI agent
                    torch.save(agent.online_actor.state_dict(), ACTOR_CHECKPOINT_FN(brain_name, agent_num))
                    torch.save(agent.online_critic.state_dict(), CRITIC_CHECKPOINT_FN(brain_name, agent_num))

        training_scores.save_scores_plot(TRAINING_SCORES_FIGURE_SAVE_PATH)
        with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
            pickle.dump(training_scores, f)

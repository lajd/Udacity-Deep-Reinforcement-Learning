from os.path import join
import pickle
import numpy as np
import torch
import torch.optim as optim
from tools.rl_constants import Brain, BrainSet
from tasks.tennis.solutions.utils import STATE_SIZE, ACTION_SIZE, BRAIN_NAME, get_simulator
from tasks.tennis.solutions.maddpg import SOLUTIONS_CHECKPOINT_DIR
from agents.maddpg_agent import MADDPGAgent
from agents.memory.memory import MemoryStreams

from agents.models.components.mlp import MLP
from agents.models.components.critics import MACritic
from tools.layer_initializations import init_layer_inverse_root_fan_in, init_layer_within_range, get_init_layer_within_rage
from simulation.utils import multi_agent_step_agents_fn, multi_agent_step_episode_agents_fn
from agents.policies.independent_maddpg_policy import IndependentMADDPGPolicy
from agents.memory.prioritized_memory import PrioritizedMemory
from tools.parameter_scheduler import ParameterScheduler
from tools.rl_constants import RandomBrainAction
from agents.models.components.noise import GaussianNoise
from agents.models.components.misc import BoundVectorNorm
from agents.models.td3 import MATD3Critic

SAVE_TAG = 'independent_madtd3'
ACTOR_CHECKPOINT_FN = lambda brain_name, agent_num: join(SOLUTIONS_CHECKPOINT_DIR, f'{brain_name}_agent_{agent_num}_{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT_FN = lambda brain_name, agent_num: join(SOLUTIONS_CHECKPOINT_DIR, f'{brain_name}_agent_{agent_num}_{SAVE_TAG}_critic_checkpoint.pth')
TRAINING_SCORES_FIGURE_SAVE_PATH_FN = lambda: join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.png')
TRAINING_SCORES_SAVE_PATH_FN = lambda: join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')

NUM_EPISODES = 10000
MAX_T = 1000
SOLVE_SCORE = 1
WARMUP_STEPS = int(1e5)
BUFFER_SIZE = int(1e6)  # replay buffer size
SEED = 0
BATCH_SIZE = 1024
BATCHNORM = True
DROPOUT = 0.1
MATD3: bool = True
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4

shared_memory = MemoryStreams(stream_ids=["TennisBrain_0", "TennisBrain_1"], capacity=BUFFER_SIZE,
                              seed=SEED)


def get_solution_brain_set():
    tennis_agents = []

    state_featurizer = MLP(
       layer_sizes=(STATE_SIZE * 2 + ACTION_SIZE, 400),
       with_batchnorm=BATCHNORM,
       activation_function=torch.nn.ReLU(True),
    )
    output_module = MLP(
        layer_sizes=(400 + ACTION_SIZE, 300, 1),
        with_batchnorm=BATCHNORM,
        activation_function=torch.nn.ReLU(True),
        output_layer_initialization_fn=get_init_layer_within_rage(limit_range=(-3e-4, 3e-4))
    )

    memory_factory = lambda: PrioritizedMemory(
        capacity=BUFFER_SIZE,
        state_shape=(1, STATE_SIZE),
        alpha_scheduler=ParameterScheduler(initial=0.6, lambda_fn=lambda i: 0.6 - 0.6 * i / NUM_EPISODES, final=0.),
        beta_scheduler=ParameterScheduler(initial=0.4, final=1,
                                          lambda_fn=lambda i: 0.4 + 0.6 * i / NUM_EPISODES),  # Anneal beta linearly
        seed=SEED,
        continuous_actions=True,
        min_priority=1e-4
    )

    if MATD3:
        critic_factory = lambda: MATD3Critic(
            critic_model_factory=lambda: MACritic(
                state_featurizer=state_featurizer,
                output_module=output_module,
                seed=SEED,
            ),
            seed=SEED
        )
    else:
        critic_factory = lambda: MACritic(
            state_featurizer=state_featurizer,
            output_module=output_module,
        )

    for i in range(2):
        key = "TennisBrain_{}".format(i)
        tennis_agent = MADDPGAgent(
            key,
            None,
            STATE_SIZE,
            ACTION_SIZE,
            critic_factory=critic_factory,
            actor_factory=lambda: MLP(
                layer_sizes=(STATE_SIZE, 400, 300, ACTION_SIZE),
                with_batchnorm=BATCHNORM,
                dropout=DROPOUT,
                output_function=BoundVectorNorm(),
                output_layer_initialization_fn=init_layer_within_range,
                hidden_layer_initialization_fn=init_layer_inverse_root_fan_in,
                seed=SEED
            ),
            critic_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=CRITIC_LR, weight_decay=1.e-5),
            actor_optimizer_factory=lambda parameters: optim.Adam(parameters, lr=ACTOR_LR),
            memory_factory=memory_factory,
            seed=0,
            batch_size=BATCH_SIZE,
            homogeneous_agents=False,
        )

        tennis_agents.append(tennis_agent)

    tennis_brain = Brain(
        brain_name=BRAIN_NAME,
        action_size=ACTION_SIZE,
        state_shape=STATE_SIZE,
        observation_type='vector',
        agents=tennis_agents,
    )

    brain_set = BrainSet(brains=[tennis_brain])

    # Update the policy with the independent MADDPG policy
    # This is done so that each agent will receive the other agents'
    # states/actions during training to guide actor learning.
    for i, agent in enumerate(tennis_agents):
        agent_id = "TennisBrain_{}".format(i)
        agent.policy = IndependentMADDPGPolicy(
            brain_set=brain_set,
            agent_id=agent_id,
            action_dim=ACTION_SIZE,
            epsilon_scheduler=ParameterScheduler(initial=1, lambda_fn=lambda i: 0.99 ** i, final=0.01),
            random_brain_action_factory=lambda: RandomBrainAction(
                ACTION_SIZE,
                1,
                continuous_actions=True,
                continuous_action_range=(-1, 1),
            ),
            map_agent_to_state_slice={
                "TennisBrain_0": lambda t: t[:, 0:24],
                "TennisBrain_1": lambda t: t[:, 24:48]
            },
            map_agent_to_action_slice={
                "TennisBrain_0": lambda t: t[:, 0:2],
                "TennisBrain_1": lambda t: t[:, 2:4]
            },
            matd3=MATD3,
            gaussian_noise_factory=lambda: GaussianNoise(),
            continuous_actions=True,
            continuous_actions_clip_range=(-1, 1)
        )

    return brain_set


if __name__ == '__main__':

    simulator = get_simulator()

    brain_set = get_solution_brain_set()

    simulator.warmup(
        brain_set,
        n_episodes=int(WARMUP_STEPS / MAX_T),
        max_t=MAX_T,
        step_agents_fn=multi_agent_step_agents_fn
    )

    brain_set, training_scores, i_episode, training_time = simulator.train(
        brain_set,
        n_episodes=NUM_EPISODES,
        max_t=MAX_T,
        solved_score=SOLVE_SCORE,
        brain_reward_accumulation_fn=lambda rewards: np.max(rewards),
        step_agents_fn=multi_agent_step_agents_fn,
        step_episode_agents_fn=multi_agent_step_episode_agents_fn
    )

    if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
        for brain_name, brain in brain_set:
            for agent_num, agent in enumerate(brain.agents):
                torch.save(agent.online_actor.state_dict(), ACTOR_CHECKPOINT_FN(brain_name, agent_num))
                torch.save(agent.online_critic.state_dict(), CRITIC_CHECKPOINT_FN(brain_name, agent_num))

        training_scores.save_scores_plot(TRAINING_SCORES_FIGURE_SAVE_PATH_FN())
        with open(TRAINING_SCORES_SAVE_PATH_FN(), 'wb') as f:
            pickle.dump(training_scores, f)

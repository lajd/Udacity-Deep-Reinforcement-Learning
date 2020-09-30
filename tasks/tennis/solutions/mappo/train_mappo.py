import pickle
import numpy as np
from os.path import join
from tools.rl_constants import Brain, BrainSet
from tasks.tennis.solutions.utils import get_simulator, STATE_SIZE, ACTION_SIZE
from tasks.tennis.solutions.mappo import SOLUTIONS_CHECKPOINT_DIR
from agents.mappo_agent import MAPPOAgent
from agents.models.ppo import MAPPO_Actor_Critic
import torch
from agents.models.components.mlp import MLP
from agents.models.components.critics import MACritic
from simulation.utils import multi_agent_step_episode_agents_fn, multi_agent_step_agents_fn
from tools.layer_initializations import init_layer_inverse_root_fan_in, get_init_layer_within_rage
from tools.parameter_scheduler import ParameterScheduler
from agents.models.components.misc import BoundVectorNorm

SAVE_TAG = 'mappo'
ACTOR_CRITIC_CHECKPOINT_FN = lambda brain_name, agent_num: join(SOLUTIONS_CHECKPOINT_DIR, f'{brain_name}_agent_{agent_num}_{SAVE_TAG}_actor_critic_checkpoint.pth')
TRAINING_SCORES_FIGURE_SAVE_PATH_FN = lambda: join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.png')
TRAINING_SCORES_SAVE_PATH_FN = lambda: join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')

NUM_EPISODES = 20000
MAX_T = 2000
SOLVE_SCORE = 1
WARMUP_STEPS = int(1e5)
SEED = 0
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPSILON = 1e-5
BATCHNORM = False
DROPOUT = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_solution_brain_set():
    tennis_agents = []
    for i in range(2):
        key = "TennisBrain_{}".format(i)
        agent = MAPPOAgent(
            agent_id=key,
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            map_agent_to_state_slice={
                "TennisBrain_0": lambda t: t[:, 0:24],
                "TennisBrain_1": lambda t: t[:, 24:48]
            },
            map_agent_to_action_slice={
                "TennisBrain_0": lambda t: t[:, 0:2],
                "TennisBrain_1": lambda t: t[:, 2:4]
            },
            actor_critic_factory=lambda: MAPPO_Actor_Critic(
                actor_model=MLP(
                    layer_sizes=(STATE_SIZE, 400, 300, ACTION_SIZE),
                    seed=SEED,
                    # output_function=BoundVectorNorm(),
                    output_function=torch.nn.Tanh(),
                    with_batchnorm=BATCHNORM,
                    activation_function=torch.nn.ReLU(True),
                    hidden_layer_initialization_fn=init_layer_inverse_root_fan_in,
                    output_layer_initialization_fn=get_init_layer_within_rage(limit_range=(-3e-4, 3e-4)),
                    dropout=DROPOUT
                ),
                critic_model=MACritic(
                    state_featurizer=MLP(
                        layer_sizes=(STATE_SIZE*2 + ACTION_SIZE, 400),
                        with_batchnorm=BATCHNORM,
                        dropout=DROPOUT,
                        seed=SEED,
                        output_function=torch.nn.ReLU(),
                    ),
                    output_module=MLP(
                        layer_sizes=(400 + ACTION_SIZE, 300, 1),
                        with_batchnorm=BATCHNORM,
                        dropout=DROPOUT,
                        seed=SEED,
                        output_layer_initialization_fn=get_init_layer_within_rage(limit_range=(-3e-4, 3e-4)),
                        activation_function=torch.nn.ReLU(True),
                    ),
                ),
                action_size=ACTION_SIZE,
                continuous_actions=True,
            ),
            optimizer_factory=lambda params: torch.optim.Adam(
                params, lr=LR, weight_decay=WEIGHT_DECAY, eps=EPSILON
            ),
            continuous_action_range_clip=(-1, 1),
            batch_size=512,
            min_batches_for_training=4,
            num_learning_updates=4,
            beta_scheduler=ParameterScheduler(initial=0.01, lambda_fn=lambda i: 0.01, final=0.01),
            std_scale_scheduler=ParameterScheduler(initial=0.8, lambda_fn=lambda i: 0.8 * 0.999 ** i, final=0.1),
            seed=SEED
        )
        tennis_agents.append(agent)

    tennis_brain = Brain(
        brain_name="TennisBrain",
        action_size=ACTION_SIZE,
        state_shape=STATE_SIZE,
        observation_type='vector',
        agents=tennis_agents,
    )

    brain_set = BrainSet(brains=[tennis_brain])
    return brain_set


if __name__ == '__main__':

    simulator = get_simulator()

    brain_set = get_solution_brain_set()

    brain_set, training_scores, i_episode, training_time = simulator.train(
        brain_set,
        n_episodes=NUM_EPISODES,
        max_t=MAX_T,
        solved_score=SOLVE_SCORE,
        step_agents_fn=multi_agent_step_agents_fn,
        step_episode_agents_fn=multi_agent_step_episode_agents_fn,
        brain_reward_accumulation_fn=lambda rewards: np.max(rewards),
        end_episode_criteria=np.all
    )

    if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
        for brain_name, brain in brain_set:
            for agent_num, agent in enumerate(brain.agents):
                torch.save(agent.online_actor_critic.state_dict(), ACTOR_CRITIC_CHECKPOINT_FN(brain_name, agent_num))

    training_scores.save_scores_plot(TRAINING_SCORES_FIGURE_SAVE_PATH_FN())
    with open(TRAINING_SCORES_SAVE_PATH_FN(), 'wb') as f:
        pickle.dump(training_scores, f)

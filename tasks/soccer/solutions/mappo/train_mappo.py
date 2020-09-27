import pickle
from os.path import join
from tools.rl_constants import Brain, BrainSet
from tasks.soccer.solutions.utils import STRIKER_STATE_SIZE, GOALIE_STATE_SIZE, NUM_STRIKER_AGENTS, \
    get_simulator, NUM_GOALIE_AGENTS, GOALIE_ACTION_SIZE, STRIKER_ACTION_SIZE, GOALIE_ACTION_DISCRETE_RANGE,\
    STRIKER_ACTION_DISCRETE_RANGE, STRIKER_BRAIN_NAME, GOALIE_BRAIN_NAME
from tasks.soccer.solutions.mappo import SOLUTIONS_CHECKPOINT_DIR
from agents.maddpg_agent import DummyMADDPGAgent
from agents.mappo_agent import MAPPOAgent
import numpy as np
from agents.models.ppo import MAPPO_Actor_Critic
import torch
from agents.models.components.mlp import MLP
from agents.models.components.critics import MACritic
from simulation.utils import multi_agent_step_episode_agents_fn, multi_agent_step_agents_fn
from tools.parameter_scheduler import ParameterScheduler


SAVE_TAG = 'mappo'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 10000
MAX_T = 2000
SOLVE_SCORE = 0.995
SEED = 0

ACTOR_CRITIC_CHECKPOINT_FN = lambda brain_name, agent_num: join(SOLUTIONS_CHECKPOINT_DIR, f'{brain_name}_agent_{agent_num}_{SAVE_TAG}_SCORE={SOLVE_SCORE}_actor_critic_checkpoint.pth')
TRAINING_SCORES_FIGURE_SAVE_PATH_FN = lambda: join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_SCORE={SOLVE_SCORE}_training_scores.png')
TRAINING_SCORES_SAVE_PATH_FN = lambda: join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_SCORE={SOLVE_SCORE}_training_scores.pkl')


"""
RED: Agent 0
Blue: Agent 1
"""


def get_solution_brain_set():
    params = {
        'striker_actor_layer_size': (STRIKER_STATE_SIZE, 256, 256, len(range(*STRIKER_ACTION_DISCRETE_RANGE))),
        'goalie_actor_layer_size': (GOALIE_STATE_SIZE, 256, 256, len(range(*GOALIE_ACTION_DISCRETE_RANGE))),
        'striker_critic_state_featurizer_layer_size': (336*4 + 3, 256),
        'striker_critic_output_layer_size': (256 + 1, 256, 1),
        'goalie_critic_state_featurizer_layer_size': (336 * 4 + 3, 256),
        'goalie_critic_output_layer_size': (256 + 1, 256, 1),
        'batchnorm': True,
        'actor_dropout': 0.1,
        'critic_dropout': 0.2,
        'lr': 5e-3,
        'weight_decay': 1e-4,
        'eps': 1e-6,
        'num_ppo_epochs': 4,
        'minimum_training_batches': 32,
        'batch_size': 1024
    }

    goalie_agents = []
    for agent_num in range(NUM_GOALIE_AGENTS):
        key = 'GoalieBrain_{}'.format(agent_num)
        if agent_num == 1:
            goalie_agent = DummyMADDPGAgent(
                GOALIE_STATE_SIZE,
                len(range(*GOALIE_ACTION_DISCRETE_RANGE)),
                seed=SEED,
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
            goalie_agent = MAPPOAgent(
                agent_id=key,
                state_size=GOALIE_STATE_SIZE,
                action_size=len(range(*GOALIE_ACTION_DISCRETE_RANGE)),
                seed=SEED,
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
                actor_critic_factory=lambda: MAPPO_Actor_Critic(
                    actor_model=MLP(
                        layer_sizes=params['goalie_actor_layer_size'],
                        seed=SEED,
                        output_function=torch.nn.Softmax(),
                        with_batchnorm=params['batchnorm'],
                        activation_function=torch.nn.LeakyReLU(True),
                        dropout=params['actor_dropout']
                    ),
                    critic_model=MACritic(
                        state_featurizer=MLP(
                            layer_sizes=params['goalie_critic_state_featurizer_layer_size'],
                            with_batchnorm=params['batchnorm'],
                            dropout=params['critic_dropout'],
                            seed=SEED
                        ),
                        output_module=MLP(
                            layer_sizes=params['goalie_critic_output_layer_size'],
                            with_batchnorm=params['batchnorm'],
                            dropout=params['critic_dropout'],
                            seed=SEED,
                        ),
                    ),
                    action_size=GOALIE_ACTION_SIZE,
                    continuous_actions=False,
                    seed=SEED
                ),
                min_batches_for_training=params['minimum_training_batches'],
                num_learning_updates=params['num_ppo_epochs'],
                optimizer_factory=lambda model_params: torch.optim.AdamW(
                    model_params, lr=params['lr'], weight_decay=params['weight_decay'], eps=params['eps']
                ),
                continuous_actions=False,
                batch_size=params['batch_size'],
                beta_scheduler=ParameterScheduler(initial=0.01, lambda_fn=lambda i: 0.01, final=0.01),
                std_scale_scheduler=ParameterScheduler(initial=0.8,
                                                       lambda_fn=lambda i: 0.8 * 0.999 ** i,
                                                       final=0.2),
            )
            print("Goalie is: {}".format(goalie_agent.online_actor_critic))
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
            striker_agent = MAPPOAgent(
                agent_id=key,
                state_size=STRIKER_STATE_SIZE,
                action_size=len(range(*STRIKER_ACTION_DISCRETE_RANGE)),
                seed=SEED,
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
                actor_critic_factory=lambda: MAPPO_Actor_Critic(
                    actor_model=MLP(
                        layer_sizes=params['striker_actor_layer_size'],
                        seed=SEED,
                        output_function=torch.nn.Softmax(),
                        with_batchnorm=params['batchnorm'],
                        activation_function=torch.nn.LeakyReLU(True),
                        dropout=params['actor_dropout']
                    ),
                    critic_model=MACritic(
                        state_featurizer=MLP(
                            layer_sizes=params['striker_critic_state_featurizer_layer_size'],
                            with_batchnorm=params['batchnorm'],
                            dropout=params['critic_dropout'],
                            seed=SEED,
                        ),
                        output_module=MLP(
                            layer_sizes=params['striker_critic_output_layer_size'],
                            with_batchnorm=params['batchnorm'],
                            dropout=params['critic_dropout'],
                            seed=SEED,
                        ),
                    ),
                    action_size=STRIKER_ACTION_SIZE,
                    continuous_actions=False,
                    seed=SEED
                ),
                optimizer_factory=lambda model_params: torch.optim.AdamW(
                    model_params, lr=params['lr'], weight_decay=params['weight_decay'], eps=params['eps']
                ),
                min_batches_for_training=params['minimum_training_batches'],
                num_learning_updates=params['num_ppo_epochs'],
                continuous_actions=False,
                batch_size=params['batch_size'],
                beta_scheduler=ParameterScheduler(initial=0.01, lambda_fn=lambda i: 0.01, final=0.01),
                std_scale_scheduler=ParameterScheduler(initial=0.8,
                                                       lambda_fn=lambda i: 0.8 * 0.999 ** i,
                                                       final=0.2),
            )
            print("Striker is: {}".format(striker_agent.online_actor_critic))
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
    return brain_set


def episode_reward_fn(brain_episode_scores):
    """ Calculate the episode reward score
    :param brain_episode_scores:
    :return:
    """
    team_scores = np.zeros(2)
    for brain_name_, agent_scores in brain_episode_scores.items():
        team_scores += agent_scores
    return 1 if np.isclose(round(team_scores[0]), 1) else 0


def end_of_episode_score_display_fn(i_episode, episode_aggregated_score, scores):
    return '\rEpisode {}\t AI Team wins {}/{} previous games'.format(
        i_episode,
        sum(scores.sliding_scores),
        len(scores.sliding_scores)
    )


def aggregate_end_of_episode_score_fn(scores):
    """ Aggregate scores over historical episodes

    Calculate the win fraction, which is the percentage \
    (over the last 100 episodes) of wins of the AI agent
    over a random agent. Note that a draw is considered a loss
    by the AI agent.
    """
    last_100_scores = scores.scores[-100:]
    win_fraction = sum(last_100_scores) / 100
    return win_fraction


if __name__ == '__main__':

    simulator = get_simulator()

    brain_set = get_solution_brain_set()
    brain_set, training_scores, i_episode, training_time = simulator.train(
        brain_set,
        n_episodes=NUM_EPISODES,
        max_t=MAX_T,
        solved_score=SOLVE_SCORE,
        episode_reward_accumulation_fn=lambda brain_episode_scores: episode_reward_fn(brain_episode_scores),
        step_agents_fn=multi_agent_step_agents_fn,
        step_episode_agents_fn=multi_agent_step_episode_agents_fn,
        end_of_episode_score_display_fn=end_of_episode_score_display_fn,
        aggregate_end_of_episode_score_fn=aggregate_end_of_episode_score_fn,
        end_episode_criteria=np.all
    )

    if training_scores.get_mean_sliding_scores() >= SOLVE_SCORE:
        for brain_name, brain in brain_set:
            for agent_num, agent in enumerate(brain.agents):
                if agent_num == 0:
                    # Only AI agent
                    torch.save(agent.online_actor_critic.state_dict(), ACTOR_CRITIC_CHECKPOINT_FN(brain_name, agent_num))
        training_scores.save_scores_plot(TRAINING_SCORES_FIGURE_SAVE_PATH_FN())
        with open(TRAINING_SCORES_SAVE_PATH_FN(), 'wb') as f:
            pickle.dump(training_scores, f)

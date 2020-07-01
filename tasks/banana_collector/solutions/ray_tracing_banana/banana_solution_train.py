from os.path import dirname, join
import torch
from agents.models.dqn import DQN
from agents.models.components.mlp import MLP
from torch import nn
from copy import deepcopy
from tasks.banana_collector.solutions.utils import default_cfg, get_policy, get_memory, get_agent, VECTOR_STATE_SHAPE, ACTION_SIZE, get_simulator
from tools.scores import Scores

SEED = default_cfg['SEED']
SOLVED_SCORE = 13.0
ENVIRONMENTS_DIR = join(dirname(dirname(dirname(__file__))), 'environments')
SOLUTION_CHECKPOINT_DIR = join(dirname(__file__), 'solution_checkpoint')
MODEL_SAVE_PATH = join(SOLUTION_CHECKPOINT_DIR, 'ray_tracing_banana_solution.pth')
PLOT_SAVE_PATH = join(SOLUTION_CHECKPOINT_DIR, 'ray_tracing_banana_solution.png')
default_cfg['N_EPISODES'] = 800


def get_solution_agent():
    params = deepcopy(default_cfg)
    update_params = {
        "MLP_FEATURES_HIDDEN": (512,),
        "OUTPUT_FC_HIDDEN_SIZES": (128,),
        "NUM_STACKED_FRAMES": 1,
        "MLP_FEATURES_DROPOUT": None,
        "OUTPUT_HIDDEN_DROPOUT": None,
        "DUELING": True,
    }

    params.update(update_params)

    policy = get_policy(ACTION_SIZE, params)

    featurizer = MLP(
        tuple([VECTOR_STATE_SHAPE[1]] + list(params['MLP_FEATURES_HIDDEN'])),
        dropout=params['MLP_FEATURES_DROPOUT'],
        activation_function=nn.ReLU(),
        output_function=nn.ReLU(),
        seed=SEED
    )

    model = DQN(
        VECTOR_STATE_SHAPE,
        ACTION_SIZE,
        featurizer,
        params['MLP_FEATURES_HIDDEN'][-1],
        seed=SEED,
        grayscale=params["GRAYSCALE"],
        num_stacked_frames=params["NUM_STACKED_FRAMES"],
        output_hidden_layer_size=params["OUTPUT_FC_HIDDEN_SIZES"],
        OUTPUT_HIDDEN_DROPOUT=params["OUTPUT_HIDDEN_DROPOUT"],
        dueling_output=params["DUELING"],
        noisy_output=params['NOISY'],
        categorical_output=params['CATEGORICAL'],
    )

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['INITIAL_LR'])
    memory = get_memory(VECTOR_STATE_SHAPE, params)
    agent_ = get_agent(VECTOR_STATE_SHAPE, ACTION_SIZE, model, policy, memory, optimizer, params)
    return agent_, params


if __name__ == '__main__':
    simulator = get_simulator(visual=False)
    agent, params = get_solution_agent()
    # Perform training
    agent, training_scores, i_episode, training_time = simulator.train(
        agents=[agent],
        solved_score=SOLVED_SCORE,
        max_t=params["MAX_T"],
        n_episodes=default_cfg['N_EPISODES'],
    )

    scores = Scores(initialize_scores=training_scores.scores)
    title_text = f"Agent scores achieving {scores.get_mean_sliding_scores()} " \
                 f"mean score in {i_episode} episodes after {round(training_time/60)}m"

    scores.save_scores_plot(PLOT_SAVE_PATH, title_text=title_text)
    fig = scores.plot_scores()

    model_save_path = join(SOLUTION_CHECKPOINT_DIR, 'ray_tracing_banana_solution.pth')

    if training_scores.get_mean_sliding_scores() > SOLVED_SCORE:
        agent.save(model_save_path)

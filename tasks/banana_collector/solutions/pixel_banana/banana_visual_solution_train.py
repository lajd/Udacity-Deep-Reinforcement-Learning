import os
import json
import torch
from os.path import join, dirname
from agents.models.dqn import VisualDQN
from agents.models.components.cnn import CNN
from copy import deepcopy
from tools.scores import Scores
from tasks.banana_collector.solutions.utils import default_cfg, get_policy, get_memory, get_agent, get_simulator, VISUAL_STATE_SHAPE, ACTION_SIZE

SEED = default_cfg['SEED']
SOLVED_SCORE = 13.0
ENVIRONMENTS_DIR = join(dirname(dirname(dirname(__file__))), 'environments')
SOLUTION_CHECKPOINT_DIR = join(dirname(__file__), 'solution_checkpoint')
MODEL_SAVE_PATH = join(SOLUTION_CHECKPOINT_DIR, 'visual_agent_banana_solution.pth')
PLOT_SAVE_PATH = join(SOLUTION_CHECKPOINT_DIR, 'visual_agent_banana_solution.png')
os.makedirs(SOLUTION_CHECKPOINT_DIR, exist_ok=True)


def get_solution_agent():
    # Define the solution hyper parameters
    params = deepcopy(default_cfg)

    update_params = {
        "INITIAL_LR": 5e-4,
        "NUM_STACKED_FRAMES": 4,
        "OUTPUT_HIDDEN_DROPOUT": 0.1,
        "DUELING": True,
        "NOISY": True,
        "BATCH_SIZE": 64,
        "N_FILTERS": (64, 128, 128),
        "EPS_DECAY_FACTOR": 0.995,
        "KERNEL_SIZES": [(1, 8, 8), (1, 4, 4), (4, 3, 3)],
        "STRIDE_SIZES": [(1, 4, 4), (1, 2, 2), (1, 3, 3)],
        "OUTPUT_FC_HIDDEN_SIZES": (1024,),
        "WARMUP_STEPS": 5000,
    }

    params.update(update_params)

    print("Params are: {}".format(json.dumps(params, indent=2)))

    policy = get_policy(ACTION_SIZE, params)

    featurizer = CNN(
        image_shape=VISUAL_STATE_SHAPE[1:],
        num_stacked_frames=params["NUM_STACKED_FRAMES"],
        grayscale=params["GRAYSCALE"],
        nfilters=params["N_FILTERS"],
        kernel_sizes=params["KERNEL_SIZES"],
        stride_sizes=params["STRIDE_SIZES"],
    )

    model = VisualDQN(
        VISUAL_STATE_SHAPE,
        ACTION_SIZE,
        featurizer,
        featurizer.output_size,
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

    memory = get_memory(VISUAL_STATE_SHAPE, params)

    agent_ = get_agent(VISUAL_STATE_SHAPE, ACTION_SIZE, model, policy, memory, optimizer, params)
    return agent_, params


if __name__ == '__main__':
    # Initialize the simulator
    simulator = get_simulator(visual=True)

    agent, params = get_solution_agent()

    # Run Training
    agent, training_scores, i_episode, training_time = simulator.train(
        agents=[agent],
        max_t=params["MAX_T"],
        solved_score=SOLVED_SCORE,
    )

    scores = Scores(initialize_scores=training_scores.scores)

    training_time = round(training_time/60)
    title_text = f"Agent scores achieving {scores.get_mean_sliding_scores()} " \
                 f"mean score in {i_episode} episodes after {training_time}m"
    scores.save_scores_plot(save_path=PLOT_SAVE_PATH, title_text=title_text)
    fig = scores.plot_scores(title_text=title_text)

    if training_scores.get_mean_sliding_scores() > SOLVED_SCORE:
        agent.save(MODEL_SAVE_PATH)

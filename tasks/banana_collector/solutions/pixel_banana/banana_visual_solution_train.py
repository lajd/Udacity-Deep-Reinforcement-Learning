import os
import json
import torch
import pickle
from os.path import join, dirname
from agents.models.dqn import VisualDQN
from agents.models.components.cnn import CNN
from copy import deepcopy
from tools.scores import Scores
from tasks.banana_collector.solutions.utils import default_cfg, get_policy, get_memory, get_agent, get_simulator, VISUAL_STATE_SHAPE, ACTION_SIZE, BRAIN_NAME
from tools.rl_constants import Experience, Brain, BrainSet
from tools.image_utils import RGBImage

SEED = default_cfg['SEED']
SOLVED_SCORE = 13.0
ENVIRONMENTS_DIR = join(dirname(dirname(dirname(__file__))), 'environments')
SOLUTION_CHECKPOINT_DIR = join(dirname(__file__), 'solution_checkpoint')
MODEL_SAVE_PATH = join(SOLUTION_CHECKPOINT_DIR, 'visual_agent_banana_solution.pth')
PLOT_SAVE_PATH = join(SOLUTION_CHECKPOINT_DIR, 'visual_agent_banana_solution.png')
TRAINING_SCORES_SAVE_PATH = join(SOLUTION_CHECKPOINT_DIR, 'solution_training_scores.pkl')
os.makedirs(SOLUTION_CHECKPOINT_DIR, exist_ok=True)


def get_solution_brain_set():
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
        "WARMUP_STEPS": 10000,
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

    solution_agent = get_agent(VISUAL_STATE_SHAPE, ACTION_SIZE, model, policy, memory, optimizer, params)

    def preprocess_state_fn(state: torch.Tensor):
        assert state.shape == torch.Size((1, 1, 84, 84, 3)), state.shape
        if params["GRAYSCALE"]:
            # bsize x num_stacked_frames x r x g x b
            image = RGBImage(state)
            # Remove the last dimension by converting to grayscale
            gray_image = image.to_gray()
            normalized_image = gray_image / 255
            preprocessed_state = normalized_image
        else:
            state /= 255
            preprocessed_state = state

        return preprocessed_state

    banana_brain_ = Brain(
        brain_name=BRAIN_NAME,
        action_size=ACTION_SIZE,
        state_shape=VISUAL_STATE_SHAPE,
        observation_type='visual',
        agent=solution_agent,
        num_agents=1,
        preprocess_state_fn=preprocess_state_fn
    )

    brain_set_ = BrainSet(brains=[banana_brain_])
    return brain_set_, params


if __name__ == '__main__':
    # Initialize the simulator
    simulator = get_simulator(visual=True)

    brain_set, params = get_solution_brain_set()

    # Run warmup
    simulator.warmup(brain_set=brain_set, n_episodes=int(params['WARMUP_STEPS'] / params['MAX_T']), max_t=params['MAX_T'])

    # Run Training
    brain_set, training_scores, i_episode, training_time = simulator.train(
        brain_set=brain_set,
        max_t=params["MAX_T"],
        solved_score=SOLVED_SCORE,
        n_episodes=default_cfg['N_EPISODES'],
    )

    scores = Scores(initialize_scores=training_scores.scores)

    training_time = round(training_time/60)
    title_text = f"Agent scores achieving {scores.get_mean_sliding_scores()} " \
                 f"mean score in {i_episode} episodes after {training_time}m"
    scores.save_scores_plot(save_path=PLOT_SAVE_PATH, title_text=title_text)
    fig = scores.plot_scores(title_text=title_text)

    if training_scores.get_mean_sliding_scores() > SOLVED_SCORE:
        brain = brain_set[BRAIN_NAME]
        trained_agent = brain.agent
        torch.save(trained_agent.online_qnetwork.state_dict(), MODEL_SAVE_PATH)
        with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
            pickle.dump(training_scores, f)

import torch
from agents.models.dqn import DQN
from agents.models.components.mlp import MLP
from torch import nn
from copy import deepcopy
from ax import optimize
import os
import pickle
import ast
from tasks.banana_collector.solutions.utils import default_cfg, get_policy, get_memory, get_agent, ACTION_SIZE, VECTOR_STATE_SHAPE, get_simulator

TRIAL_COUNTER = 0
NUM_TRIALS = 500
SEED = default_cfg['SEED']
default_cfg['N_EPISODES'] = 500
SOLVED_SCORE = 13.0
TUNINGS_DIR = os.path.abspath('ray_tunings')
os.makedirs(TUNINGS_DIR, exist_ok=True)


def write_tuning_data(info: dict, performance: float):
    with open(f'{TUNINGS_DIR}/trial_{TRIAL_COUNTER}_performance_{performance}', 'wb') as f:
        pickle.dump(info, f)


def banana_tuning(update_params: dict):
    params = deepcopy(default_cfg)
    params.update(update_params)
    try:
        params['OUTPUT_FC_HIDDEN_SIZES'] = ast.literal_eval(params['OUTPUT_FC_HIDDEN_SIZES'])
        params['SUPPORT_RANGE'] = ast.literal_eval(params['SUPPORT_RANGE'])
        params['MLP_FEATURES_HIDDEN'] = ast.literal_eval(params['MLP_FEATURES_HIDDEN'])

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

        agent = get_agent(VECTOR_STATE_SHAPE, ACTION_SIZE, model, policy, memory, optimizer, params)

        # Run performance evaluation
        performance, info = simulator.get_agent_performance(
            agent=agent,
            n_train_episodes=params["N_EPISODES"],
            n_eval_episodes=params["N_EVAL_EPISODES"],
            max_t=params["MAX_T"],
        )
        info['input_params'] = params

        write_tuning_data(info, performance)

        global TRIAL_COUNTER
        TRIAL_COUNTER += 1

        print("Performance is : {}".format(performance))
        return performance
    except Exception as e:
        print(e)
        return 0


if __name__ == "__main__":
    # Initialize the simulator
    simulator = get_simulator(visual=False)
    best_parameters, best_values, _, _ = optimize(
        # https://ax.dev/docs/core.html#search-space-and-parameters
        parameters=[
            {"name": "INITIAL_LR",
             "type": "range",
             "bounds": [5e-6, 5e-3]
             },
            {"name": "GAMMA",
             "type": "range",
             "bounds": [0.8, 0.999]
             },
            {"name": "ACTION_REPEATS",
             "type": "choice",
             "values": [1, 2, 3, 4]
             },
            {"name": "UPDATE_FREQUENCY",
             "type": "choice",
             "values": [4, 16, 64]
             },
            {"name": "MLP_FEATURES_HIDDEN",
             "type": "choice",
             "values": ['(64, 64)', '(128, 128)', '(512, 512)', '(512, )']
             },
            {"name": "OUTPUT_FC_HIDDEN_SIZES",
             "type": "choice",
             "values": ['(64, 64)', '(128, 64)', '(64, 32)', '(512, )']
             },
            {"name": "OUTPUT_HIDDEN_DROPOUT",
             "type": "choice",
             "values": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
             "value_type": 'float',
             },
            {"name": "CATEGORICAL",
             "type": "choice",
             "values": [True, False]
             },
            {"name": "SUPPORT_RANGE",
             "type": "choice",
             "values": ['(-10, 10)', '(-50, 50)', '(-100, 100)']
             },
            {"name": "NOISY",
             "type": "choice",
             "values": [True, False]
             },
            {"name": "DUELING",
             "type": "choice",
             "values": [True, False]
             },
        ],
        evaluation_function=banana_tuning,
        minimize=False,
        arms_per_trial=1,
        total_trials=NUM_TRIALS
    )
    print("Best parameters::: {}".format(best_parameters))

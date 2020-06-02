from agents.models.dqn import VisualDQN
from agents.models.components.cnn import CNN
import torch
from copy import deepcopy
from ax import optimize
import os
import pickle
import ast

from tasks.banana_collector.solutions.utils import default_cfg, get_policy, get_memory, get_agent, VISUAL_STATE_SHAPE, ACTION_SIZE, get_simulator

SEED = default_cfg['SEED']
TUNINGS_DIR = os.path.abspath('visual_tunings')
os.makedirs(TUNINGS_DIR, exist_ok=True)
TRIAL_COUNTER = 0
NUM_TRIALS = 500  # Some will error

# Update default params for visual DQN
default_cfg['N_EPISODES'] = 800
default_cfg['NUM_STACKED_FRAMES'] = 4
default_cfg['WARMUP_STEPS'] = 5000


def write_tuning_data(info: dict, performance: float):
    with open(f'{TUNINGS_DIR}/trial_{TRIAL_COUNTER}_performance_{performance}', 'wb') as f:
        pickle.dump(info, f)


def visual_banana_tuning(update_params: dict):
    params = deepcopy(default_cfg)
    params.update(update_params)
    try:
        params['SUPPORT_RANGE'] = ast.literal_eval(params['SUPPORT_RANGE'])
        params['OUTPUT_FC_HIDDEN_SIZES'] = ast.literal_eval(params['OUTPUT_FC_HIDDEN_SIZES'])
        params['N_FILTERS'] = ast.literal_eval(params['N_FILTERS'])
        params['KERNEL_SIZES'] = [ast.literal_eval(i) for i in ast.literal_eval(params["KERNEL_SIZES"])]
        params['STRIDE_SIZES'] = [ast.literal_eval(i) for i in ast.literal_eval(params["STRIDE_SIZES"])]

        policy = get_policy(ACTION_SIZE, params)
        print(params)
        featurizer = CNN(
            state_shape=VISUAL_STATE_SHAPE,
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

        agent = get_agent(VISUAL_STATE_SHAPE, ACTION_SIZE, model, policy, memory, optimizer, params)

        # Run performance evaluation
        performance, info = simulator.get_agent_performance(
            agent=agent,
            n_train_episodes=params["N_EPISODES"],
            n_eval_episodes=params["N_EVAL_EPISODES"],
            max_t=params["MAX_T"],
        )
        info['input_params'] = params

        global TRIAL_COUNTER
        TRIAL_COUNTER += 1

        write_tuning_data(info, performance)

        print(f"Performance is : {performance}")
        return performance
    except Exception as e:
        # Failures can occur do to invalid CNN sizes
        print(e)
        return 0


if __name__ == '__main__':
    simulator = get_simulator(visual=True)

    best_parameters, best_values, _, _ = optimize(
        # https://ax.dev/docs/core.html#search-space-and-parameters
        parameters=[
            {"name": "INITIAL_LR",
             "type": "range",
             "bounds": [5e-4, 1e-3]
             },
            {"name": "GAMMA",
             "type": "range",
             "bounds": [0.8, 0.999]
             },
            {"name": "N_FILTERS",
             "type": "choice",
             "values": [
                 '(128, 128, 256)',
                 '(64, 128, 256)',
                 '(128, 256, 256)',
             ]
             },
            {"name": "OUTPUT_FC_HIDDEN_SIZES",
             "type": "choice",
             "values": ['(256, 256)', '(128, 256)', '(512, 256)', '(1024, 256)', '(2048, 256)', '(2048,)', ]
             },
            {
                "name": "KERNEL_SIZES",
                "type": "choice",
                "values": [
                    "('(1, 8, 8)', '(1, 4, 4)', '(4, 3, 3)')",
                    "('(1, 8, 8)', '(1, 4, 4)', '(4, 1, 1)')",
                    "('(1, 8, 8)', '(1, 3, 3)', '(4, 3, 3)')",
                    "('(1, 8, 8)', '(1, 5, 5)', '(4, 3, 3)')",
                ],
            },
            {
                "name": "STRIDE_SIZES",
                "type": "choice",
                "values": [
                    "('(1, 3, 3)', '(1, 3, 3)', '(1, 3, 3)')",
                    "('(1, 4, 4)', '(1, 2, 2)', '(1, 1, 1)')",
                    "('(1, 5, 5)', '(1, 3, 3)', '(1, 3, 3)')",
                    "('(1, 4, 4)', '(1, 2, 2)', '(1, 3, 3)')",
                ],
            },
            {"name": "OUTPUT_HIDDEN_DROPOUT",
             "type": "choice",
             "values": [0, 0.1, 0.2],
             "value_type": 'float',
             },
            {"name": "CATEGORICAL",
             "type": "choice",
             "values": [True, False]
             },
            {"name": "SUPPORT_RANGE",
             "type": "choice",
             "values": ['(-10, 20)', '(-10, 10)']
             },
            {"name": "NOISY",
             "type": "choice",
             "values": [True, False]
             },
            {"name": "DUELING",
             "type": "choice",
             "values": [True, False]
             },
            # {"name": "GRAYSCALE",
            #  "type": "choice",
            #  "values": [True, False]
            #  },
        ],
        evaluation_function=visual_banana_tuning,
        minimize=False,
        arms_per_trial=1,
        total_trials=NUM_TRIALS
    )

    print("Best parameters::: {}".format(best_parameters))

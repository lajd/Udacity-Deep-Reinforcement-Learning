import torch
from agents.policies.epsilon_greedy import EpsilonGreedyPolicy
from agents.dqn_agent import DQNAgent
from agents.policies.categorical import CategoricalDQNPolicy
from agents.policies.max import MaxPolicy
from agents.memory.prioritized_memory import Memory
from tools.parameter_decay import ParameterScheduler
from unityagents import UnityEnvironment
from simulation.unity_environment import UnityEnvironmentSimulator
from os.path import join, dirname

ENVIRONMENTS_DIR = join(dirname(dirname(__file__)), 'environments')
VISUAL_STATE_SHAPE = (1, 84, 84, 3)
VECTOR_STATE_SHAPE = (1, 37)
ACTION_SIZE = 4

default_cfg = {
    ###############
    # Common params
    ###############
    "SEED": 123,
    "INITIAL_LR": 5e-4,
    "N_EPISODES": 1000,
    "MAX_T": 1000,
    "TAU": 1e-3,  # for soft update of target parameters,
    "GAMMA": 0.99,  # discount factor,
    "NUM_STACKED_FRAMES": 1,
    "BATCH_SIZE": 32,
    "ACTION_REPEATS": 1,
    "UPDATE_FREQUENCY": 4,
    "WARMUP_STEPS": 1000,
    "LR_GAMMA": 0.995,
    "OUTPUT_FC_HIDDEN_SIZES": (128,),
    "OUTPUT_HIDDEN_DROPOUT": None,
    ###############
    # Architecture
    #############
    # Categorical DQN
    "CATEGORICAL": False,
    "NUM_ATOMS": 51,
    "SUPPORT_RANGE": (-10, 10),
    # Noisy DQN
    "NOISY": False,
    # Dueling DQN
    "DUELING": False,
    ##############
    # Preprocessing
    ##############
    "GRAYSCALE": False,
    ###############
    # Policy params
    ###############
    "EPS_DECAY_FACTOR": 0.995,
    "FINAL_EPS": 0.01,
    ################
    # Featurizers
    ###############
    # CNN Featurizer
    "N_FILTERS": (32, 64, 64),
    "KERNEL_SIZES": [(1, 3, 3), (1, 3, 3), (4, 3, 3)],
    "STRIDE_SIZES": [(1, 3, 3), (1, 3, 3), (1, 3, 3)],
    # MLP Featurizer params
    "MLP_FEATURES_HIDDEN": (128,),
    "MEMORY_CAPACITY": int(5e4),
    "MLP_FEATURES_DROPOUT": None,
    #########
    # Tuning
    #########
    "N_EVAL_EPISODES": 10,
}


def get_simulator(visual: bool = False):
    if visual:
        observation_type = 'visual'
        environment_name = "VisualBanana_Linux/Banana.x86_64"
    else:
        observation_type = 'vector'
        environment_name = "Banana_Linux/Banana.x86_64"
    # Initialize the simulator
    env = UnityEnvironment(file_name=join(ENVIRONMENTS_DIR, environment_name))
    simulator = UnityEnvironmentSimulator(
        task_name='{}_banana_collector'.format(observation_type),
        env=env, seed=default_cfg["SEED"],
        observation_type=observation_type
    )
    return simulator


def get_policy(action_size: int, params):
    if params['CATEGORICAL']:
        policy = CategoricalDQNPolicy(
            action_size=action_size,
            num_atoms=params["NUM_ATOMS"],
            v_min=params['SUPPORT_RANGE'][0],
            v_max=params['SUPPORT_RANGE'][1],
            seed=params['SEED']
        )
    elif params['NOISY']:
        policy = MaxPolicy(
            action_size=action_size
        )
    else:
        decay_factor = params['EPS_DECAY_FACTOR']
        final_eps = params['FINAL_EPS']
        policy = EpsilonGreedyPolicy(
            action_size=action_size,
            epsilon_scheduler=ParameterScheduler(initial=1, lambda_fn=lambda i: decay_factor**i, final=final_eps),
        )

    return policy


def get_memory(state_shape: tuple, params):
    memory = Memory(
        capacity=params['MEMORY_CAPACITY'],
        state_shape=state_shape,
        num_stacked_frames=params["NUM_STACKED_FRAMES"],
        alpha_scheduler=ParameterScheduler(initial=0.6, lambda_fn=lambda i: 0.6),
        beta_scheduler=ParameterScheduler(initial=0.4, final=1,
                                          lambda_fn=lambda i: 0.4 + 0.6 * i / params["N_EPISODES"]),
        seed=params['SEED']
    )
    return memory


def get_agent(state_shape: tuple, action_size: int, model: torch.nn.Module, policy, memory, optimizer, params):
    # Agent
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        model=model,
        policy=policy,
        batch_size=params['BATCH_SIZE'],
        update_frequency=params['UPDATE_FREQUENCY'],
        gamma=params['GAMMA'],
        warmup_steps=params['WARMUP_STEPS'],
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params['LR_GAMMA']),
        optimizer=optimizer,
        memory=memory,
        seed=params['SEED'],
        tau=params['TAU'],
        action_repeats=params['ACTION_REPEATS'],
    )
    return agent

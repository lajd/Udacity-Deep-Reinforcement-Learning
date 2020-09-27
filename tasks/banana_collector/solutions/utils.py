import torch
from agents.policies.epsilon_greedy import EpsilonGreedyPolicy
from agents.dqn_agent import DQNAgent
from agents.policies.categorical_policy import CategoricalDQNPolicy
from agents.policies.max_policy import MaxPolicy
from agents.memory.prioritized_memory import ExtendedPrioritizedMemory
from tools.parameter_scheduler import ParameterScheduler
from unityagents import UnityEnvironment
from simulation.unity_environment import UnityEnvironmentSimulator
from os.path import join, dirname
from tools.lr_schedulers import DummyLRScheduler
from tools.image_utils import RGBImage
from typing import Callable
from tools.rl_constants import BrainSet, Experience
import numpy as np


ENVIRONMENTS_DIR = join(dirname(dirname(__file__)), 'environments')
IMAGE_SHAPE = (84, 84, 3)
VISUAL_STATE_SHAPE = (1, 84, 84, 3)
VECTOR_STATE_SHAPE = (1, 37)
ACTION_SIZE = 4
BRAIN_NAME = 'BananaBrain'

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
    "FILTERS": (32, 64, 64),
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
    # memory = Memory(buffer_size=int(1e6), seed=6)
    memory = ExtendedPrioritizedMemory(
        capacity=params['MEMORY_CAPACITY'],
        state_shape=state_shape,
        num_stacked_frames=params["NUM_STACKED_FRAMES"],
        alpha_scheduler=ParameterScheduler(initial=0.6, lambda_fn=lambda i: 0.6 - 0.6 * i / params['N_EPISODES'], final=0.0),
        beta_scheduler=ParameterScheduler(initial=0.4, final=1, lambda_fn=lambda i: 0.4 + 0.6 * i / params["N_EPISODES"]),
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
        lr_scheduler=DummyLRScheduler(optimizer),  # Using adam
        optimizer=optimizer,
        memory=memory,
        seed=params['SEED'],
        tau=params['TAU'],
        action_repeats=params['ACTION_REPEATS'],
    )
    return agent


def get_preprocess_state_fn(params) -> Callable:
    def preprocess_state_fn(state: torch.Tensor):
        # Environment gives extra dimension
        # print("Original state shape in preprocess_state_fn is: {}".format(state.shape))
        state = state.squeeze(0)

        assert state.shape == torch.Size((1, 84, 84, 3)), state.shape
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

    return preprocess_state_fn


def visual_agent_step_agents_fn(brain_set: BrainSet, next_brain_environment: dict, t: int):
    for brain_name, brain_environment in next_brain_environment.items():
        agent = brain_set[brain_name].agents[0]
        state = brain_environment['states']
        # state = state[np.newaxis, ...]

        # print("State shape is::: {}".format(state.shape))
        # print("Action in step agents fn is: {}".format(brain_environment['actions']))
        # print("next_states in step agents fn is: {}".format(brain_environment['next_states'].shape))

        brain_agent_experience = Experience(
            state=state,
            action=brain_environment['actions'][0],
            reward=brain_environment['rewards'],
            next_state=brain_environment['next_states'],
            done=torch.LongTensor(brain_environment['dones']),
            t_step=t,
        )
        agent.step(brain_agent_experience)

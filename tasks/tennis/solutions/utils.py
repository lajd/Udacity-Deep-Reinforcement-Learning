import os
from unityagents import UnityEnvironment
from simulation.unity_environment import UnityEnvironmentSimulator
from os.path import join, dirname


ENVIRONMENTS_DIR = join(dirname(dirname(__file__)), 'environments')

STATE_SIZE = 24
ACTION_SIZE = 2
NUM_AGENTS = 2
SEED = 0
BRAIN_NAME = 'TennisBrain'


def get_simulator():
    observation_type = 'vector'
    environment_name = "Tennis_Linux/Tennis.x86_64"
    # Initialize the simulator
    env = UnityEnvironment(file_name=join(ENVIRONMENTS_DIR, environment_name))
    simulator = UnityEnvironmentSimulator(
        task_name='{}_multi_agent_tennis'.format(observation_type),
        env=env, seed=SEED,
    )
    return simulator

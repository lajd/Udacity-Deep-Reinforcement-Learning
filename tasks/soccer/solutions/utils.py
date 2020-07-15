from unityagents import UnityEnvironment
from simulation.unity_environment import UnityEnvironmentSimulator
from os.path import join, dirname

ENVIRONMENTS_DIR = join(dirname(dirname(__file__)), 'environments')
NUM_GOALIE_AGENTS = 2
NUM_STRIKER_AGENTS = 2
GOALIE_STATE_SIZE = 336  # 112 * 3 stacked frames
STRIKER_STATE_SIZE = 336  # 112 * 3 stacked frames
GOALIE_ACTION_SIZE = 1
STRIKER_ACTION_SIZE = 1
GOALIE_ACTION_DISCRETE_RANGE = (0, 3)
STRIKER_ACTION_DISCRETE_RANGE = (0, 5)
GOALIE_BRAIN_NAME = 'GoalieBrain'
STRIKER_BRAIN_NAME = 'StrikerBrain'
SEED = 0


def get_simulator():
    observation_type = 'vector'
    environment_name = "Soccer_Linux/Soccer.x86_64"
    # Initialize the simulator
    env = UnityEnvironment(file_name=join(ENVIRONMENTS_DIR, environment_name))
    simulator = UnityEnvironmentSimulator(
        task_name='{}_multi_agent_goalie_striker_soccer'.format(observation_type),
        env=env, seed=SEED,
        observation_type=observation_type
    )
    return simulator

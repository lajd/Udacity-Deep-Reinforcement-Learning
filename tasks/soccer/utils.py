from unityagents import UnityEnvironment
from simulation.unity_environment import UnityEnvironmentSimulator
from os.path import join, dirname

ENVIRONMENTS_DIR = join(dirname(dirname(__file__)), 'environments')
NUM_GOALIE_AGENTS = 2
NUM_STRIKER_AGENTS = 6
GOALIE_STATE_SIZE = 336
STRIKER_STATE_SIZE = 336
SEED = 0


def get_simulator():
    observation_type = 'vector'
    environment_name = "Soccer_Linux/Soccer.x86_64"
    # Initialize the simulator
    env = UnityEnvironment(file_name=join(ENVIRONMENTS_DIR, environment_name))
    simulator = UnityEnvironmentSimulator(
        task_name='{}_multi_agent_mixed_game_type_soccer'.format(observation_type),
        env=env, seed=SEED,
        observation_type=observation_type
    )
    return simulator

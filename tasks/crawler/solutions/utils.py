from unityagents import UnityEnvironment
from simulation.unity_environment import UnityEnvironmentSimulator
from os.path import join, dirname

ENVIRONMENTS_DIR = join(dirname(dirname(__file__)), 'environments')
STATE_SIZE = 129
ACTION_SIZE = 20
NUM_AGENTS = 12
SEED = 0
BRAIN_NAME = 'CrawlerBrain'


def get_simulator():
    observation_type = 'vector'
    environment_name = "Crawler_Linux/Crawler.x86_64"
    # Initialize the simulator
    env = UnityEnvironment(file_name=join(ENVIRONMENTS_DIR, environment_name))
    simulator = UnityEnvironmentSimulator(
        task_name='{}_crawler_continuous_control'.format(observation_type),
        env=env, seed=SEED,
    )
    return simulator

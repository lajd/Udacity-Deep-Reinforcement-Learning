from unityagents import UnityEnvironment
from simulation.unity_environment import UnityEnvironmentSimulator
from os.path import join, dirname

ENVIRONMENTS_DIR = join(dirname(dirname(__file__)), 'environments')
STATE_SIZE = 33
ACTION_SIZE = 4
SEED = 0


def get_simulator():
    observation_type = 'vector'
    environment_name = "Reacher_Linux/Reacher.x86_64"
    # Initialize the simulator
    env = UnityEnvironment(file_name=join(ENVIRONMENTS_DIR, environment_name))
    simulator = UnityEnvironmentSimulator(
        task_name='{}_robotic_arm_continuous_control'.format(observation_type),
        env=env, seed=SEED,
        observation_type=observation_type
    )
    return simulator

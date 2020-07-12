from os.path import dirname, join
from tasks.banana_collector.solutions.ray_tracing_banana.banana_solution_train import MODEL_SAVE_PATH, get_solution_brain_set
from tasks.banana_collector.solutions.utils import default_cfg, get_simulator, BRAIN_NAME

SEED = default_cfg['SEED']
ENVIRONMENTS_DIR = join(dirname(dirname(dirname(__file__))), 'environments')

if __name__ == '__main__':
    # Initialize the simulator
    simulator = get_simulator(visual=False)

    brain_set, params = get_solution_brain_set()

    brain_set[BRAIN_NAME].agent.load(MODEL_SAVE_PATH)
    simulator.evaluate(brain_set)

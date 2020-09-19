from os.path import dirname, join
from tasks.banana_collector.solutions.pixel_banana.banana_visual_solution_train import MODEL_SAVE_PATH, get_solution_brain_set, BRAIN_NAME
from tasks.banana_collector.solutions.utils import default_cfg, get_simulator

SEED = default_cfg['SEED']
ENVIRONMENTS_DIR = join(dirname(dirname(dirname(__file__))), 'environments')

if __name__ == '__main__':
    # Initialize the simulator
    simulator = get_simulator(visual=True)

    brain_set, params = get_solution_brain_set()

    brain_set[BRAIN_NAME].agents[0].load(MODEL_SAVE_PATH)
    simulator.evaluate(brain_set)

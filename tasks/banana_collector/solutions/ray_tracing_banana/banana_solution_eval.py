from tasks.banana_collector.solutions.ray_tracing_banana.banana_solution_train import MODEL_SAVE_PATH, get_solution_brain_set
from tasks.banana_collector.solutions.utils import get_simulator, BRAIN_NAME

if __name__ == '__main__':
    # Initialize the simulator
    simulator = get_simulator(visual=False)

    brain_set, params = get_solution_brain_set()

    brain_set[BRAIN_NAME].agents[0].load(MODEL_SAVE_PATH)

    brain_set, average_score = simulator.evaluate(
        brain_set,
        n_episodes=10,
    )

import torch
from tasks.crawler.solutions.utils import get_simulator
from tasks.crawler.solutions.ddpg.train_td3 import get_solution_brain_set, ACTOR_CHECKPOINT_PATH, MAX_T


if __name__ == '__main__':
    simulator = get_simulator()

    brain_set = get_solution_brain_set()

    # All agents are homogeneous so we only have to load the model for a single agent
    ref_agent = brain_set['CrawlerBrain'].agents[0]
    ref_agent.online_actor.load_state_dict(torch.load(ACTOR_CHECKPOINT_PATH))

    brain_set, average_score = simulator.evaluate(brain_set, n_episodes=10, max_t=MAX_T)

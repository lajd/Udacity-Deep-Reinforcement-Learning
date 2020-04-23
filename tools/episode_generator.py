import numpy as np


class EpsilonGreedyEpisodeGenerator:
    def __init__(self, env):
        self.env = env

        self.unit_state_vector = np.ones(self.env.nA)
        self.actions = np.arange(self.env.nA)

    def generate_episode_from_Q(self, Q, eps):
        # This method creates an episode from policy
        episode = []
        state = self.env.reset()
        while True:
            if state in Q:
                greedy_action = np.argmax(Q[state])
                probs = self.unit_state_vector * eps / self.env.nA
                probs[greedy_action] += (1 - eps)
                assert np.isclose(sum(probs), 1), sum(probs)
                action = np.random.choice(self.actions, 1, p=probs)[0]
            else:
                action = np.random.choice(np.arange(self.env.nA), 1)[0]
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

from typing import Optional, List
from collections import deque
import numpy as np
import pandas as pd
from pylab import *


class Scores:
    """ Helper class for maintaining the scores (rewards) accumulated by an agent """
    def __init__(self, tag: str = 'Training', window_size: int = 100, initialize_scores: Optional[List[float]] = None):
        if initialize_scores:
            self.scores = initialize_scores
            self.sliding_scores = deque(maxlen=window_size)
            self.sliding_scores.extend(initialize_scores)
        else:
            self.scores = []
            self.sliding_scores = deque(maxlen=window_size)

        self.tag = tag
        self.window_size = window_size

    def add(self, score: float):
        self.scores.append(score)
        self.sliding_scores.append(score)

    def get_mean_sliding_scores(self):
        return np.mean(self.sliding_scores)

    def get_plot(self, title_text: str = 'Scores (Rewards)', xlabel_text: str = 'Episode #', ylabel_txt: str = 'Score', body_txt: str = None, txt_size: str = 'x-small'):
        data = pd.DataFrame(self.scores)
        rolling_data = data.rolling(window=self.window_size).mean()

        fig = figure()
        gca().set_position((.1, .3, .8, .6))  # to make a bit of room for extra text
        scatter(data.index, data, color='blue', marker='+', label='Episode scores')
        plot(rolling_data.index, rolling_data, color='red', label="{} episode average score".format(self.window_size))
        title(title_text, wrap=True)
        ylabel(ylabel_txt)
        xlabel(xlabel_text)
        margins(0.1)
        legend()
        grid(True)
        plot([1, 2], [3, 4])
        figtext(.02, .1, body_txt, size=txt_size, wrap=True)
        return fig

    def plot_scores(self, title_text: Optional[str] = 'Scores (Rewards)'):
        # Scatter plot the scores and overlay the rolling mean
        # and the rolling mean
        plt_ = self.get_plot(title_text=title_text)
        plt_.show()

    def save_scores_plot(self, save_path: str, title_text: Optional[str] = 'Scores (Rewards)'):
        plt_ = self.get_plot(title_text=title_text)
        plt_.savefig(save_path)

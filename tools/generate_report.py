import os
import pickle
from tools import scores
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tasks.banana_collector.solutions.utils import default_cfg
from tools.scores import Scores


def get_performance(file_path: str, performance_threshold: float = 10):
    try:
        with open(file_path, 'rb') as f:
            info = pickle.load(f)
        scores_ = Scores(initialize_scores=info['train_scores'])
        mean_score = scores_.get_mean_sliding_scores()
        if mean_score > performance_threshold:
            return scores_.get_mean_sliding_scores()
    except Exception as e:
        # Corrupted binary
        pass


def save_report(project_title: str, report_save_dir: str = 'REPORT.pdf', tunings_directory: str = 'tunings', performance_threshold: float = 10.0):
    """ Generate a report based on hyperparameter tunings """
    sorted_paths = []
    for fp in [os.path.join(tunings_directory, i) for i in os.listdir(tunings_directory)]:
        performance = get_performance(fp, performance_threshold)
        if performance:
            sorted_paths.append((fp, performance))
    sorted_paths = [i[0] for i in sorted(sorted_paths, key=lambda x: x[1], reverse=True)]
    with PdfPages(report_save_dir) as pdf:
        # Create title page
        cover_page = plt.figure(figsize=(8, 8))
        cover_page.clf()
        cover_page.text(0.5, 0.9, project_title, transform=cover_page.transFigure, size=24, ha="center")

        body = 'In this report we summarize the performance of various flavours \n' \
               'of the DQN algorithm. ALl experiments use the following default \n' \
               'hyperparameter configurations. The parameters provided with each \n' \
               'figure overwrite the default parameters, differentiating each trial \n'
        cover_page.text(0.5, 0.75, body, transform=cover_page.transFigure, size=12, ha="center")

        default_params_as_txt = ''
        for i, (k, v) in enumerate(default_cfg.items(), start=1):
            default_params_as_txt += '{}={} \n'.format(k, v)

        cover_page.text(0.5, 0.1, default_params_as_txt, transform=cover_page.transFigure, size=10, ha="center", wrap=True)

        pdf.savefig()

        for fp in sorted_paths:
            try:
                with open(fp, 'rb') as f:
                    trial_info = pickle.load(f)
                trial_scores = scores.Scores(tag="Training", initialize_scores=trial_info['train_scores'])
                trial_params = trial_info['input_params']
                n_train_episodes = trial_info['n_train_episodes']
                train_time = trial_info['train_time']
                # Filter out the default parameters
                trial_params = {k: v for k, v in trial_params.items() if v != default_cfg[k]}
                txt = ''
                for i, (k, v) in enumerate(trial_params.items(), start=1):
                    txt += '{}={}; '.format(k, v)
                    if i % 5 == 0:
                        txt += '\n'
                plt_ = trial_scores.get_plot(
                    title_text=f"Agent episode scores achieving {trial_scores.get_mean_sliding_scores()} "
                               f"mean score in {n_train_episodes} episodes after {train_time}s",
                    xlabel_text="# Episodes",
                    ylabel_txt="Episode scores",
                    body_txt=txt
                )
                plt_.savefig(pdf,  format="pdf")
            except Exception as e:
                # Nothing saved
                print(e)
                pass

import torch
import torch.nn as nn
import torch.optim as optim
import os
from simulation.openai_gym_parallel_mc import ParallelMonteCarloGymEnvironment
from agents.reinforce_agent import ReinforceAgent
from agents.policies.reinforce_policy import ReinforcePolicy
from agents.models.components.cnn import CNN
from agents.models.components.mlp import MLP
from agents.models.base import BaseModel
from tools.parameter_decay import ParameterScheduler
from tasks.pong.utils import preprocess_batch, evaluate_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 1000
LR_GAMMA = 1
INITIAL_LR = 1e-4
PARALLEL_COUNT = 8
BETA_DECAY_GAMMA = 0.995
FINAL_BETA = 1e-6
INITIAL_BETA = 0.01
GAMMA = 0.99
TMAX = 320

NUM_STACKED_FRAMES = 2
featurizer = CNN(image_shape=(80, 80), num_stacked_frames=NUM_STACKED_FRAMES, grayscale=True, filters=(4, 16), kernel_sizes=(6, 6), stride_sizes=(2,4))
print("Featurizer output size is::: {}".format(featurizer.output_size))
output_layer = MLP(layer_sizes=(featurizer.output_size,  256, 1), activation_function=nn.ReLU(), output_function=nn.Sigmoid(), dropout=0.2, seed=123)


class PongModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            featurizer,
            output_layer
        )

    def forward(self, state: torch.Tensor, act: bool = True) -> torch.Tensor:
        return self.model(state)


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    model = PongModel().to(device)

    print(model)

    # model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)

    policy = ReinforcePolicy(
        beta_scheduler=ParameterScheduler(initial=INITIAL_BETA, lambda_fn=lambda i: INITIAL_BETA*BETA_DECAY_GAMMA**i, final=FINAL_BETA),
        gamma=GAMMA,
    )


    agent = ReinforceAgent(
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=LR_GAMMA),
        model=model,
        optimizer=optimizer,
        policy=policy,
    )

    simulator = ParallelMonteCarloGymEnvironment(env_name='PongDeterministic-v4', task_name='pong', parallel_count=PARALLEL_COUNT, seed=123, custom_preprocess_trajectories=preprocess_batch)

    simulator.train(agent, num_episodes=NUM_EPISODES, tmax=TMAX, num_stacked_frames=NUM_STACKED_FRAMES)

    agent.save(os.path.join(dirname, 'reinforce_agent.pt'))

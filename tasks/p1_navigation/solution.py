import torch
from unityagents import UnityEnvironment
from simulation.unity_environment import UnityEnvironmentSimulator
from agents.policies.epsilon_greedy import EpsilonGreedyPolicy
from agents.dqn_agent import DQNAgent
from tools.model_components.mlp import MLP
from tools.lr_scheduler import LRScheduler

SEED = 123
INITIAL_LR = 5e-4
N_EPISODES = 2000
MAX_T = 1000

env = UnityEnvironment(file_name="tasks/p1_navigation/Banana_Linux/Banana.x86_64")
# env = UnityEnvironment(file_name="tasks/p1_navigation/Banana_Linux_NoVis/Banana.x86_64")
simulator = UnityEnvironmentSimulator(task_name='banana_collector', env=env, seed=SEED)


policy = EpsilonGreedyPolicy(
    action_size=simulator.action_size,
    initial_eps=1,
    epsilon_decay_fn=lambda i: 0.995,
    final_eps=0.01
)

model = MLP((simulator.state_size, 512, 512, simulator.action_size), layer_dropout=0.4, output_layer=None)

optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)

lr_scheduler = LRScheduler(
    initial_lr=INITIAL_LR,
    lambda_fn=lambda i_episode: 0.995,
    optimizer=optimizer
)

agent = DQNAgent(
    state_size=simulator.state_size,
    action_size=simulator.action_size,
    model=model,
    policy=policy,
    batch_size=64,
    update_frequency=4,
    gamma=0.999,
    warmup_steps=int(10e3),
    lr_scheduler=lr_scheduler,
    optimizer=optimizer
)

# Train
agent, train_scores = simulator.train(agent=agent, solved_score=13.0, n_episodes=N_EPISODES, max_t=MAX_T)


agent, eval_scores = simulator.evaluate(agent=agent, n_episodes=N_EPISODES, max_t=MAX_T)


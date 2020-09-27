[scores]: solution_checkpoint/ppo_training_scores.png "PPO Baseline Results"

# Crawler
Please see the [repository overview](../../../../README.md) as well as the [task description](../../TASK_DETAILS.md)
before reading this report. The theoretical details of the utilized algorithms can be found in the [repository overview](../../../../README.md).

In this task there are 12 crawler agents who's goal is to reach a static location in the environment as fast as possible
(i.e. minimize falling and maximize for speed).

<img src="https://github.com/lajd/drl_toolbox/blob/master/tasks/crawler/solutions/ppo/solution_checkpoint/trained_crawler_agent.gif?raw=true" width="400" height="250" />

# Solution Overview

The solutions discussed in this report rely on the PPO algorithm. All 12 algorithms share the same PPO brain
(actor-critic and optimizer) and the same shared trajectory buffer. During training, agents may perform
batch learning by sampling from the shared replay buffer. After a small number of learning epochs, the experience
samples are discarded.

The actor-critic architecture has the following form:

```
PPO_Actor_Critic(
  (actor): MLP(
    (mlp_layers): Sequential(
      (0): BatchNorm1d(129, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Linear(in_features=129, out_features=128, bias=True)
      (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): LeakyReLU(negative_slope=True)
      (4): Linear(in_features=128, out_features=128, bias=True)
      (5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): LeakyReLU(negative_slope=True)
      (7): Linear(in_features=128, out_features=20, bias=True)
      (8): Tanh()
    )
  )
  (critic): MLP(
    (mlp_layers): Sequential(
      (0): BatchNorm1d(129, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Linear(in_features=129, out_features=128, bias=True)
      (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): LeakyReLU(negative_slope=True)
      (4): Linear(in_features=128, out_features=128, bias=True)
      (5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): LeakyReLU(negative_slope=True)
      (7): Linear(in_features=128, out_features=1, bias=True)
      (8): Tanh()
    )
  )
)
```
The model hyper-parameters are given below:

```
NUM_EPISODES = 3000
SEED = 8
MAX_T = 2000
WEIGHT_DECAY = 1e-4
EPSILON = 1e-5  # epsilon of Adam
LR = 1e-4  # learning rate of the actor-critic
BATCH_SIZE = 1024
DROPOUT = None
BATCHNORM = True
SOLVE_SCORE = 1600
```

## Results

Below we show the plot of mean episode scores (across all agents) versus episode number.

![Training scores][scores]

The environment was solved (mean reward of >=1600) after about 320 episodes.
The training time took roughly 1.3 hours.

## Discussion
The PPO algorithm demonstrated good stability and convergence, and was experimentally shown to be rather robust to changes
in hyperparameters.

## Ideas for Future Work
The MAPPO algorithm demonstrated quick convergence on this task, however it's sample efficiency leaves much to be desired. 
In order to increase the sample efficiency, memory replay methods such as [Hindsight Experience Replay (HER)](https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf)  can be implemented, 
which better help the agent learn from sparse rewards.

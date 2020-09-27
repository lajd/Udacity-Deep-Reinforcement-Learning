[trained_soccer]:https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[mappo_results_image]: solutions/mappo/solution_checkpoint/mappo_100_consecutive_wins_training_scores.png "MAPPO Training"

# Soccer MAPPO/MATD3 Introduction
Please see the [repository overview](../../README.md) as well as the [task description](./TASK_DETAILS.md)
before reading this report. The theoretical details of the utilized algorithms can be found in the [repository overview](../../README.md).

In this environment, two teams (each with a Striker/Goalie agent) compete against each other in the game of soccer. The agents can move laterally
and vertically, and the strikers have the additional action of rotating left/right, resulting in 4 and 6 discrete actions for
the Goalie and Striker, respectively.

The task is episodic, where each episode terminates when either 1) either team has scored a goal, or 2) the maximum time step has been reached. 
When a team scores a goal, they win, and when the maximum time step is reached, a draw occurs.

The goal of this task is to train a single team (eg. the red team) of agents to beat the opposing team (eg. the blue team)
95/100 times over a the previous 100 episodes. This requires the AI team to score 95/100 times -- draws are counted against
the AI team.

The unity environment consists of 4 agents which have separate brains (models/optimizers),
but can observe the states and actions of the other agents (on both their team and their opponent's team)
and use this information during training time, but not during evaluation time.

![Trained Agent][trained_soccer]

# Solution Overview

This solution discusses the MAPPO model for solving this task.
We attempted a solution with MATD3, however the model failed to reach convergence. To accommodate the MATD3 model for 
a discrete action space, target-policy-smoothing was not implemented (i.e. no noise was added to the target action).

#### Independent brains

The agents contain separate actor-critic models, optimizers, and trajectory buffers, but have access to
`joint_states` and `joint_actions` attributes, which they can manipulate to select their own
attributes (states/actions) and the attributes of other agents.

### MAPPO Solution

Goalie Actor-Critic
```
MAPPO_Actor_Critic(
  (actor): MLP(
    (mlp_layers): Sequential(
      (0): BatchNorm1d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Linear(in_features=336, out_features=256, bias=True)
      (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): LeakyReLU(negative_slope=True)
      (4): Dropout(p=0.1, inplace=False)
      (5): Linear(in_features=256, out_features=128, bias=True)
      (6): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): LeakyReLU(negative_slope=True)
      (8): Dropout(p=0.1, inplace=False)
      (9): Linear(in_features=128, out_features=4, bias=True)
      (10): Softmax(dim=None)
    )
  )
  (critic): MACritic(
    (state_featurizer): MLP(
      (mlp_layers): Sequential(
        (0): BatchNorm1d(1347, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=1347, out_features=256, bias=True)
        (2): ReLU(inplace=True)
      )
    )
    (output_module): MLP(
      (mlp_layers): Sequential(
        (0): BatchNorm1d(257, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=257, out_features=128, bias=True)
        (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): Dropout(p=0.2, inplace=False)
        (5): Linear(in_features=128, out_features=1, bias=True)
      )
    )
  )
)
```


Striker Actor-Critic 
```
Striker is: MAPPO_Actor_Critic(
  (actor): MLP(
    (mlp_layers): Sequential(
      (0): BatchNorm1d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Linear(in_features=336, out_features=256, bias=True)
      (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): LeakyReLU(negative_slope=True)
      (4): Dropout(p=0.1, inplace=False)
      (5): Linear(in_features=256, out_features=128, bias=True)
      (6): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): LeakyReLU(negative_slope=True)
      (8): Dropout(p=0.1, inplace=False)
      (9): Linear(in_features=128, out_features=6, bias=True)
      (10): Softmax(dim=None)
    )
  )
  (critic): MACritic(
    (state_featurizer): MLP(
      (mlp_layers): Sequential(
        (0): BatchNorm1d(1347, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=1347, out_features=256, bias=True)
        (2): ReLU(inplace=True)
      )
    )
    (output_module): MLP(
      (mlp_layers): Sequential(
        (0): BatchNorm1d(257, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=257, out_features=128, bias=True)
        (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): Dropout(p=0.2, inplace=False)
        (5): Linear(in_features=128, out_features=1, bias=True)
      )
    )
  )
)
```

##### Results

Below we show the score plot for the AI team over time, where the score is determined by the win-fraction over the 
previous 100 episodes. Thus, a reward of 0.5 indicates that the AI team has won (i.e. not lost or achieved a draw) 50/100
of the previous episodes, and a reward of 1 indicates that the AI team has won all of the previous 100 episodes.

![Training MAPPO Agent][mappo_results_image]

The MAPPO agents (AI team) was able to beat the random team 100 times consecutively after ~5100 epsiodes, 

##### Discussion
The MAPPO algorithm showed good convergence while solving this task, whereas the MATD3 algorithm
did not show appreciable convergence.

## Ideas for Future Work
The MAPPO algorithm demonstrated quick convergence on this task, however it's sample efficiency leaves much to be desired. 
In order to increase the sample efficiency, memory replay methods such as [Hindsight Experience Replay (HER)](https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf)  can be implemented, 
which better help the agent learn from sparse rewards.
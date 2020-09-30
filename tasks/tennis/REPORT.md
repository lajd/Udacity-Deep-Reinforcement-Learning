[trained_tennis_gif]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[mappo_results_image]: solutions/mappo/solution_checkpoint/mappo_training_scores.png "MAPPO Training"
[matd3_results_image]: solutions/maddpg/solution_checkpoint/independent_madtd3_training_scores.png "MATD3 Training"

# Tennis MAPPO/MATD3 Introduction
Please see the [repository overview](../../../../README.md) as well as the [task description](./TASK_DETAILS.md)
before reading this report. The theoretical details of the utilized algorithms can be found in the [repository overview](../../../../README.md).

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

The unity environment consists of 2 agents which have separate brains (models/optimizers),
but can observe the states and actions of the other agents and use this information during training time.


![Trained Agent][trained_tennis_gif]

# Solution Overview

This solution discusses the MAPPO and the MATD3 models for solving this task.

#### Independent brains

The agents contain separate actor-critic models, optimizers, and trajectory buffers, but have access to
`joint_states` and `joint_actions` attributes, which they can manipulate to select their own
attributes (states/actions) and the attributes of other agents.

### MAPPO Solution

The MAPPO actor-critic:
```
MAPPO_Actor_Critic(
  (actor): MLP(
    (mlp_layers): Sequential(
      (0): Linear(in_features=24, out_features=400, bias=True)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=400, out_features=300, bias=True)
      (3): ReLU(inplace=True)
      (4): Linear(in_features=300, out_features=2, bias=True)
      (5): Tanh()
    )
  )
  (critic): MACritic(
    (state_featurizer): MLP(
      (mlp_layers): Sequential(
        (0): Linear(in_features=50, out_features=400, bias=True)
        (1): ReLU(inplace=True)
        (2): ReLU()
      )
    )
    (output_module): MLP(
      (mlp_layers): Sequential(
        (0): Linear(in_features=402, out_features=300, bias=True)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=300, out_features=1, bias=True)
      )
    )
  )
)

```

With model hyper-parameters:

```
NUM_EPISODES = 20000
MAX_T = 2000
SOLVE_SCORE = 1
WARMUP_STEPS = int(1e5)
SEED = 0
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPSILON = 1e-5
BATCHNORM = False
DROPOUT = None
```

### MATD3 Solution

The MATD3 actor:

```
MLP(
  (mlp_layers): Sequential(
    (0): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Linear(in_features=24, out_features=400, bias=True)
    (2): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Dropout(p=0.1, inplace=False)
    (5): Linear(in_features=400, out_features=300, bias=True)
    (6): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU(inplace=True)
    (8): Dropout(p=0.1, inplace=False)
    (9): Linear(in_features=300, out_features=2, bias=True)
    (10): BoundVectorNorm()
  )
)
```

The MATD3 critic:

```
MATD3Critic(
  (q_network_a): MACritic(
    (state_featurizer): MLP(
      (mlp_layers): Sequential(
        (0): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=50, out_features=400, bias=True)
        (2): ReLU(inplace=True)
      )
    )
    (output_module): MLP(
      (mlp_layers): Sequential(
        (0): BatchNorm1d(402, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=402, out_features=300, bias=True)
        (2): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): Linear(in_features=300, out_features=1, bias=True)
      )
    )
  )
  (q_network_b): MACritic(
    (state_featurizer): MLP(
      (mlp_layers): Sequential(
        (0): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=50, out_features=400, bias=True)
        (2): ReLU(inplace=True)
      )
    )
    (output_module): MLP(
      (mlp_layers): Sequential(
        (0): BatchNorm1d(402, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=402, out_features=300, bias=True)
        (2): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): Linear(in_features=300, out_features=1, bias=True)
      )
    )
  )
)

```

With model hyper-parameters:

```
NUM_EPISODES = 10000
MAX_T = 1000
SOLVE_SCORE = 1
WARMUP_STEPS = int(1e5)
BUFFER_SIZE = int(1e6)  # replay buffer size
SEED = 0
BATCH_SIZE = 1024
BATCHNORM = True
DROPOUT = 0.1
MATD3: bool = True
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4
```
##### Results

Below we show the reward plot, obtained by averaging over the agents' shared rewards (blue) over 100 episodes (red).
The episodic reward is computed by taking the maximum reward between the two agents at each episode. The MAPPO 
algorithm achieves a score of >0.5 in ~ 7500 episodes (15 minutes), and a score of > 1 in about 3200 episodes (18 minutes)

###### Results for MAPPO:
![Training MAPPO Agent][mappo_results_image]


###### Results for MATD3
![Training MATD3 Agent][matd3_results_image]


##### Discussion
The MAPPO algorithm converged *significantly* faster than the MATD3 algorithm, achieving a score of >1 about 33x faster
than the MAPPO algorithm (20 minutes vs. 11 hours). It should be noted, though, that hyper-parameter tuning 
(especially on the MATD3 algorithm) was not conducted due to the long training duration. Overall, this result demonstrates 
the robustness of the PPO algorithm to a wide range of tasks.

The MAPPO algorithm, beign on-policy, is shown to be relatively sample inefficient compared to off-policy algorithms such as 
MATD3, where MAPPO achieved a score of > 1 after 3200 episodes compared to 800 episodes by MATD3. The MATD3 algorithm takes
advantage of prioritized experience replay (PER) to sample experience based on the amount of information the experience provides, while
the MAPPO algorithm has no such intelligent memory buffer and simply discards trajectories of experience after a few learning epochs.


## Ideas for Future Work
The MAPPO algorithm demonstrated quick convergence on this task, however it's sample efficiency leaves much to be desired. 
In order to increase the sample efficiency, memory replay methods such as [Hindsight Experience Replay (HER)](https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf)  can be implemented, 
which better help the agent learn from sparse rewards.
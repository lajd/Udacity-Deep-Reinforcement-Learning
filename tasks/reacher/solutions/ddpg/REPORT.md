[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: resources/ddpg_baseline.png "DDPG Baseline Results"
[image3]: resources/per_td3_baseline.png "TD3 PER Baseline Results"

# Reacher (Continuous Control)
Please see the [repository overview](../../../../README.md) as well as the [task description](../../TASK_DETAILS.md)
before reading this report. The theoretical details of the utilized algorithms can be found in the [repository overview](../../../../README.md).

The Unity environment contains two versions:
 - Single agent version
 - 20 agent version (each agent experiences it's own copy of the environment)

In the solution provided here, we only address the 20-agent version, although the models developed can easily be 
adapted to the single agent version as well. The 20-agent version was preferred since 1) it allows for multiple 
independent experience streams to be sampled from the environment, resulting in a more robust training dataset,
and 2) the multi-agent environment is suitable for distributed algorithms such as PPO, A3C and D4PG. Unfortunately
these algorithms were not developed in this solution, but will be saved for a later implementation.


![Trained Agent][image1]

# Solution Overview

The solutions discussed in this report rely on the DDPG algorithm and it's TD3 variant.

In the 20-agent environment, the experiences from each agent are collected and stored in the same memory replay buffer. 
All agents share the same brain (actor, critic and optimizer), and each agent can sample from the shared memory
buffer and perform batch learning. In this way, the purpose of having multiple agents is just to sample multiple
trajectories of experience from the environment for a single model to learn over. The model components are as follows:

- Memory
    - (Prioritized) replay buffer shared across all agents
    - Agents can sample from the experiences other agents have collected
- Online Actor
    - Online actor model
- Target Actor
    - Target actor model
- Actor Optimizer
    - Optimizer for the actor network
- Online Critic
    - Online critic model
- Target Critic
    - Target critic model
- Critic Optimizer
    - Optimizer for the critic
- Policy
    - Contains logic for selecting actions and computing actor/critic losses
    - Options are: 
        - DDPG Policy
        - TD3 Policy
- Actor Optimizer Scheduler
    - Learning rate scheduler for the actor optimizer
- Critic Optimizer Scheduler
    - Learning rate scheduler for the critic optimizer

## TD3 Solution

##### Setup and Architecture
The TD3 actor/critic models are simple MLPs with leaky_relu activations as shown below:

- Actor
```
TD3Actor(
  (fc1): Linear(in_features=33, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
)
```
- Critic
```
TD3Critic(
  (q_network_a): Critic(
    (fc1): Linear(in_features=33, out_features=256, bias=True)
    (fc2): Linear(in_features=260, out_features=128, bias=True)
    (fc3): Linear(in_features=128, out_features=1, bias=True)
  )
  (q_network_b): Critic(
    (fc1): Linear(in_features=33, out_features=256, bias=True)
    (fc2): Linear(in_features=260, out_features=128, bias=True)
    (fc3): Linear(in_features=128, out_features=1, bias=True)
  )
)
```

Where the major difference between TD3 and DDPG is that the critic has been split into two independent streams, each learning an independent 
estimate of the q-value. See the [TD3 section](../../../../README.md) of the readme for details of the TD3 architecture.

- Memory
    - A prioritized replay buffer was utilized, demonstrating a faster learning rate than the standard replay buffer
    - A warmup of 5000 steps per agent is conducted with random actions to initialize the replay buffer
    - &alpha; is annealed linearly to 0 from an initial value of 0.6
    - &beta; is annealed linearly to 1 from an initial value of 0.4
    
- Noise (used both for exploration and for evaluating actions)
    - Gaussian noise was used for both exploration and action evaluation. The use of Gaussian noise led to 
      slightly better convergence than the Ornstein-Uhlenbeck process.

Leaky relu activations were chosen as they demonstrated slightly better convergence and stability as compared to
ReLus.

The model hyper-parameters are given below:

```
NUM_AGENTS = 20
NUM_EPISODES = 200
SEED = 0
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = int(1e6)
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
N_LEARNING_ITERATIONS = 10     # number of learning updates
UPDATE_FREQUENCY = 20       # every n time step do update
MAX_T = 1000
CRITIC_WEIGHT_DECAY = 0.0  # 1e-2
ACTOR_WEIGHT_DECAY = 0.0
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
POLICY_UPDATE_FREQUENCY = 2
WARMUP_STEPS = int(5e3)
MIN_PRIORITY = 1e-3
```

The TD3 algorithm makes use of the POLICY_UPDATE_FREQUENCY parameter, which indicates after how many time steps
the policy network and target-network updates should be applied. 


##### Results

Below we show the plot of mean episode scores (across all agents) versus episode number.

![Trained Agent][image3]

The environment was solved (mean reward of >=30) after 61 episodes. 
The training time was substantial at roughly 1.5 hours. In contrast, the vanilla DDPG algorithm with non-prioritized
replay demonstrated poor convergence and increased sensitivity to hyperparameter choices (results for DDPG not
shown as a result).

##### Discussion
The TD3 algorithm demonstrated good stability, convergence and sample efficiency properties, solving the environment in 
just 61 episodes. The task was learned very quickly, where the "solved" score of 30 was achieved in 
only about 10 episodes, where another 50 episodes were required to bring the average score to > 30. 

We note that the TD3 algorithm performed better using Gaussian noise for exploration and action evaluation rather than 
the Ornstein-Uhlenbeck process, and the use of the prioritized replay buffer improved the initial learning rate as 
compared to the vanilla replay buffer.

## Ideas for Future Work
There are a number of important extensions and ideas for future work, which we discuss below:
###### Robust hyper parameter and architecture tuning
The performance of TD3 was found to be sensitive to hyper parameters and model architectures. 
For example, changing the initial learning rate, the noise-distribution parameters and the number
of hidden layers significantly. Previously for the Banana-Collector task we took advantage of [Pytorch's Ax package](https://ax.dev/) for
hyper parameter tuning, and doing the same analysis here would be useful. One challenge preventing the usefulness of hyper parameter tuning
is the length of time it takes to train in these environments. For this reason, it would be useful to develop heuristic measures of performance 
which do not require significant training time, looking at features such as stability and learning rate.

###### Experiment with alternative architectures
1) Implement [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf) and contrast performance with 
prioritized experience replay and the standard replay buffer
2) Utilize sequential information by stacking frames, utilizing recurrent networks and performing action repeats
3) Experiment with alternative noise strategies for exploration and action evaluation
4) Experiment with layer normalization and regularization techniques such as batch/layer normalization and dropout
5) Implementation and experimentation of N-step/multi-step returns 

###### Implementation of distributed algorithms
The time required for training was one of the main bottlenecks for development, making asynchronous/parallel 
algorithms appealing. This becomes more important when we wish to perform intensive tasks such as hyper parameter tuning.

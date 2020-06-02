[image1]: solution_checkpoint/ray_tracing_banana_solution.png "Episode Scores"
[image2]: ../../resources/banana_collector/alternative_dqn.png "Episode Scores"

# Ray-Tracing Banana Collector

Please see the [repository overview](../../../../README.md) before reading this report.

## Method
The solution model architecture was determined through automated hyperparameter tuning 
using Pytorch's [Ax library](https://ax.dev/). This can be run by executing the script
[hyperparameter_tuning.py](hyperparameter_tuning.py), which saves model artifacts into the directory [ray_tunings](ray_tunings).

The results of the tuning can be automatically extracted into a report using the [generate_report.py](generate_report.py)
script, which iterates over tuning artifacts and creates a PDF report profiling the reward-vs-episode
profiles for various flavours of models.

## Solution model architecture
The best-The ray-tracing banana task was solved using the following model architecture: </br>
```
DQN(
  (features): MLP(
    (mlp_layers): Sequential(
      (0): Linear(in_features=37, out_features=512, bias=True)
      (1): ReLU()
    )
  )
  (output): OutputLayer(
    (advantage): MLP(
      (mlp_layers): Sequential(
        (0): Linear(in_features=512, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=4, bias=True)
      )
    )
    (value): MLP(
      (mlp_layers): Sequential(
        (0): Linear(in_features=512, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=1, bias=True)
      )
    )
  )
)

```
which is uses the DQN network with the following extensions: </br>
- `Featurizer`
    - The MLP featurizer was selected with a single hidden layer of size (512), with no dropout.
    - ReLu nonlinearities was used
- `Dueling networks`
    - Dueling networks were utilized with an output hidden layer of size 128 and no dropout.
    - ReLu nonlinearities were used

The selected model configuration is as follows: </br>

```
"MLP_FEATURES_HIDDEN": (512,)
"OUTPUT_FC_HIDDEN_SIZES": (128,)
"NUM_STACKED_FRAMES": 1
"MLP_FEATURES_DROPOUT": None
"OUTPUT_HIDDEN_DROPOUT": None
"EPS_DECAY_FACTOR": 0.995
"FINAL_EPS": 0.01
"DUELING": True
"MEMORY_CAPACITY": int(5e4)
"MLP_FEATURES_DROPOUT": None
```



###### Model
The [MLP featurizer](../../../../agents/models/components/mlp.py) is utilized for extraction features
from the raw state vector, and consists of a single hidden layer of size 512 with ReLu activation. No dropout
layers were applied to the MLP featurizer.

The output of the network uses a dueling architecture provided to the [DQN](../../../../agents/models/dqn.py) module
as a flag, with a hidden layer size of 128 with no dropout. 

We note that the selected model did not consist of the categorical DQN or noisy DQN enhancements.

Only a single frame was used (no stacking of previous frames was performed).

###### Policy
Because the Noisy DQN flavour was not used, the policy was a simple &epsilon;-greedy policy which
was decayed multiplicatively by a factor of 0.995<sup>i_episode</sub>, with an initial value of 1 and
a minimum value of 0.01.

###### Memory
The memory module implements a [prioritized experience replay buffer](../../../../agents/memory/prioritized_memory.py), 
which stores a finite amount of experience, and samples in a non-uniform way according to the priority of the experience.
See the [Prioritized Experience Replay Paper](https://arxiv.org/abs/1511.05952) for more information, or the [the code](../../../../agents/memory/prioritized_memory.py)
for the implementation.

### Results: 
Below we show the scores for each episode (blue), and the average score over the last 100 episodes (red), until our agent 
has solved the task (average score of 13 over the past 100 episodes) for the architecture and hyper-parameters listed
above. 

The agent solves the task in 511 episodes in 16 minutes.

![Trained Agent][image1]

### Discussion: 
From the supplementary [RESULTS.pdf](RESULTS.pdf), we can see that a range of hyper-parameter/architectures are able to 
perform well on the tasks, and given more trials, it is likely that we would find more performant models. The model architectures
each offer their own unique long-term performance for this task, as well as other attributes such as episode score variance, 
learning plateaus, etc. 

Below we show the scores plot for network which takes advantage of both the noisy and categorical DQN enhancements.

![Trained Agent][image2]

In the first plot, the network only has the Dueling DQN enhancement and uses an epsilon-greedy policy with multiplicative 
decay by a factor of 0.995<sup>i_episode</sup>. In contrast, in the second plot the network uses dueling, noisy and categorical enhancements
and uses the [Max Policy](../../../../agents/policies/max.py), which simply selects the action with the highest value of Q(s<sub>t+1</sub>, a<sub>t+1</sub>).
We can see that the two plots show very different properties, with the first showing a more gradual but steady improvement over time, 
while the second shows a slower start and a steeper performance improvement followed by an early plateau. The variances of the scores
is also significantly different, with the first network showing relatively low variance in scores over the coarse of the task, and the
second showing increased variance as it progresses. In this way, we can see that the space of possible models has a sophisticated interaction
with an agent's performance in a given environment. 

Interestingly, the best-performing model does not take advantage of all of the improvements to the original
DQN algorithm, namely the `noisy output` and `categorical` DQN enhancements. In contrast, the best-performing model for
the [visual DQN](../pixel_banana/REPORT.md) was shown to utilise the noisy output layer. 

### Future work
- A more thorough investigation as to the cost/benefits of using NoisyDQN, Dueling DQN, DoubleDQN, Prioritized Experience Replay
 and other extensions towards the Rainbow algorithm. It's important to understand in which situations these algorithms bring result
 in better performance
- Have a better way of intuiting how the algorithms are learning. For example, the Dueling DQN architecture provides the value V(s)
  for each state. Overlaying the value map V(s) against the environment would help provide intuition as to how the network is learning
  (for example, the value of being close to a yellow banana should be high). Confirming that the network's learning is in line with
  expectations would act as both a sanity check and deeper insights. Another example would be the return distributions output
  by the categorical DQN algorithm. In my preliminary experiments, the distribution was highly skewed to one action, causing
  it to be selected most frequently, and often resulting in poor results (for example, the agent taking entirely left turns throughout an episode).
  I noticed that the likelihood for this depended on the model's capacity. Having a more intuitive understanding for how the model is utilizing
  categorical distributions would help the practitioner (me) better understand how to engineer the models.
- Make the hyper-parameter tuning module more robust. There are a wide range of hyperparameters and model architectures to experiment with, and having
  performance data of the different combinations would be very informative. The [Ax](https://ax.dev/tutorials/) library offers much functionality which
  I was not able to try out during this project, but could be instrumental for developing RL agents in a streamline and automated fashion, for a range of 
  different environments.
- Incorporate open source solutions, such as [Coach](https://github.com/NervanaSystems/coach), which offer high-performance and state of the art implementations
  of these models.

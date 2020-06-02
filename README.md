# DRL Toolbox

## Overview
This repository contains code reinforcement learning code for solving the 
Udacity Deep reinforcement Learning projects.

## Prerequisites
- Anaconda
- Linux with GPU
- CUDA 10.2

All models are developed in Pytorch
## Install
Recreate the Anaconda environment with: <br/> 
`conda env create -f environment.yml`

## Repository Structure
The code is organized as follows:

- [/agents](agents)
    - Agents are responsible for interacting with the environment -- given a state provided from
    the environment, the agent should be able to choose an action. At the next time step the agent
    will receive a reward for that action. The goal of the agent is to choose actions in order to 
    maximize the long-term (cumulative) reward.
    
    - [/agents/models](agents/models):
        - Models are interchangeable components representing an agent's brain. The current implementations
        of models are parametric and invoke neural networks.
        - The input to a model is a `State`, and the output is a representation Q(s<sub>t</sub>, a<sub>t</sub>; &theta;)
        of the value of the (s, a) pair
        
        - [/agents/models/components](agents/models/components):
            - Model components are `torch.nn.Modules` which are combined to form models. In the context of 
            the DQN model, these components form the featurizer and outputs of the DQN model. 
            
    - [/agents/policies](agents/policies):
        - Policies are responsible for choosing the next action given the current state. Policies utilize the parametric models
          from [/agents/models](agents/models) for the purpose of generating actions given a state. 
        - The `Policy` classes are responsible for taking Q(a<sub>t</sub>, s<sub>t</sub>), 
        generated from a `Model` and outputting an `Action`
    
    - [/agents/memory](agents/memory):
        - Implements memory buffers for the agent, used for storing experiences sampled from the environment and recalling
          experiences during learning
           
- [/simulation](simulation)
    - The simulation modules are responsible for emulating the the environment, and provide helper functions for training
    and evaluating an agent. The simulator provides environmental state for the agent, receives an
    action, and generates the next environmental state along with a scalar reward signal for the agent.

- [/tasks](tasks)
    - Tasks are the learning tasks (or problems) which we develop DRL agents to solve. 
    - `/tasks/<task_name>`
        - The solution and discussion for a particular task can be found in these folders
    
- [/tools](tools)
    - Contains common tools for solving tasks

## Task solutions and reports
### Yellow Banana Collector
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
![Trained Agent][image1]
#### Introduction
In this task, the agent's goal is to collect as many yellow bananas as possible while avoiding blue bananas. </br>
    There are two variations to this task -- the `Ray-Tracing Banana Collector`, where the state from the environment
    is a small (37) dimensional vector of hand-crafted features of the agent's velocity, along with ray-based perception
    of objects around agent's forward direction. In contrast, the `Pixel-Based Banana Collector` receives a raw pixel (RGB image)
    as state from the environment. Below we go into more details on the task description: </br>

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.
#### Action Space
For both the `Ray-based` and `Pixel-based` tasks, the actions available to the agent are the same:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

At each time step, the agent must provide an action to the environment.

##### Reward Structure
For both the `Ray-based` and `Pixel-based` tasks, the reward structure is the same as well.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

##### State Space: Ray Tracing Banana Collector
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

##### State Space: Pixel-Based Banana Collector
The state space is a 3D tensor of size (84, 84, 3) representing an RGB image of width=height=84 pixels. 


#### Preparing the Unity ML-Agent environments

If you're running on linux and wish to download both environments, you can navigate to the [task directory](tasks/banana_collector) run the task directory script [setup_linux.sh](tasks/banana_collector/setup_linux.sh).
If you're running on a different OS, please follow the instructions below for each environment.

##### Ray-Tracing Banana Collector

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in `/tasks/banana_collector/environments/Banana_Linux`, and unzip (or decompress) the file. 

##### Pixel-based Banana Collector
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `/tasks/banana_collector/environments/VisualBanana_Linux`, and unzip (or decompress) the file. 

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.

#### Quick navigation
- Ray-tracing implementation
    - [ray tracing banana collector results](tasks/banana_collector/solutions/ray_tracing_banana/RESULTS.pdf)
    - [ray tracing banana collector report](tasks/banana_collector/solutions/ray_tracing_banana/REPORT.md)
    - [train_best_model](tasks/banana_collector/solutions/ray_tracing_banana/banana_solution_train.py)
    - [eval_best_model](tasks/banana_collector/solutions/ray_tracing_banana/banana_solution_eval.py)
- Visual (pixel) implementation
    - [pixel based banana collector report](tasks/banana_collector/solutions/pixel_banana/REPORT.md)
    - [train_best_model](tasks/banana_collector/solutions/pixel_banana/banana_visual_solution_train.py)
    - [eval_best_model](tasks/banana_collector/solutions/pixel_banana/banana_visual_solution_train.py)

## Agent Implementations and explanation
Currently only the [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) algorithm is implemented, along with
a few extensions to the original algorithm outlined in the [Rainbow DQN](https://arxiv.org/abs/1710.02298) implementation.
Below, we discuss the algorithm at a high level, along with the implemented extensions. </br>
 - [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (DQN)
    ###### Overview
    [Q-Learning](https://en.wikipedia.org/wiki/Q-learning#Variants) is a value-based reinforcement learning algorithm which can be
    used to learn an optimal policy (strategy) for selection actions in an environment. Being value based, the Q-learning algorithm
    aims to identify the expected return (cumulative sum of future rewards): </br>
     
    Q : S x A ->  &#8477;, for all s &in; S, a &in; A
    
    Q<sup>&pi;</sup>(s<sub>t</sub>, a<sub>t</sub>) = &expectation;[R<sub>t</sub>, &gamma;R<sub>t+1</sub>, &gamma;<sup>2</sup>R<sub>t+2</sub> + ... | s<sub>t</sub>, a<sub>t</sub>]
    
    Where &pi; is the policy being followed, t is the current time step, and R<sub>t</sub> is the reward (provided by the environment) at time t.
    
    The goal in Q-learning is to find the optimal action-value function, or: </br>
    
    Q<sup>*</sup>(s<sub>t</sub>, a<sub>t</sub>) = max<sub>&pi;</sub> &expectation;[R<sub>t</sub>, &gamma;R<sub>t+1</sub>, &gamma;<sup>2</sup>R<sub>t+2</sub> + ... | s<sub>t</sub>, a<sub>t</sub>]
    
    The optimal action value function thus the maximum expected return which can be achieved by taking action a in state s.
    
    The optimal action-value function obeys the Bellman equation, which allows Q<sup>*</sup>(s<sub>t</sub>, a<sub>t</sub>) to be writting as a recurrence relation:
    
    Q<sup>*</sup>(s<sub>t</sub>, a<sub>t</sub>) = &expectation;[R<sub>t</sub> + &gamma; max<sub>a<sub>t+1</sub></sub> Q<sup>*</sup>(s<sub>t+1</sub>, a<sub>t+1</sub>) | s<sub>t</sub>, a<sub>t</sub>]
    
    Which can then be cast as an iterative update rule as: 
        
    Q<sub>i+1</sub>(s<sub>t</sub>, a<sub>t</sub>) = &expectation;<sub>s<sub>t+1</sub></sub>[R<sub>t</sub> + &gamma; max<sub>a<sub>t+1</sub></sub> Q<sub>i</sub>(s<sub>t+1</sub>, a<sub>t+1</sub>) | s<sub>t</sub>, a<sub>t</sub>]
    
    Where i indicates the update step. The [Deep Q-Network Algorithm](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) 
    uses parametric (neural network) models as function approximators to estimate the value of Q(s, a), resulting in iterative loss equations of:
    
    L<sub>i</sub>(&theta;<sub>i</sub>) = &expectation;<sub>s<sub>t</sub>,a<sub>t</sub>,r<sub>t</sub></sub> [&expectation;<sub>s<sub>t+1</sub></sub>([R<sub>t</sub> + &gamma; max<sub>a<sub>t+1</sub></sub> Q<sub>i</sub>(s<sub>t+1</sub>, a<sub>t+1</sub>; &theta;<sup>-</sup>)] - Q(s<sub>t</sub>, a<sub>t</sub>; &theta;<sub>i</sub>))<sup>2</sup>]
    
    Where the target network's weights &theta;<sup>-</sup> represents a set of parameters which are periodically copied (synced) with teh online network's weights, &theta;, every &tau; time steps, and held fixed otherwise. 
    For convenience, we can write the target value function as:
    
    Y<sub>t</sub><sup>DQN</sup> &equiv; R<sub>t</sub> + &gamma; max<sub>a<sub>t+1</sub></sub> Q(s<sub>t+1</sub>, a<sub>t+1</sub>; &theta;<sub>t</sub><sup>-</sup>)
    
    ###### [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
      The [double DQN](https://arxiv.org/pdf/1509.06461.pdf) algorithm is a minor but significant adaptation to the DQN 
      algorithm which helps reduce over-estimation of action values Q(s, a). The traditional DQN algorithm has: 
      
      Y<sub>t</sub><sup>DQN</sup> &equiv; R<sub>t</sub> + &gamma; max<sub>a<sub>t+1</sub></sub> Q(s<sub>t+1</sub>, a<sub>t+1</sub>; &theta;<sub>t</sub><sup>-</sup>)
       = R<sub>t</sub> + &gamma;Q(S<sub>t+1</sub>, argmax<sub>a</sub>Q(S<sub>t+1</sub>, a;&theta;<sub>t</sub>); &theta;<sub>t</sub>)
      
      where the same &theta;<sub>t</sub> is used to compute both the current state-values and next state-values, where we've used the fact
      that a<sub>t+1</sub> is approximated by argmax<sub>a</sub>(Q(S<sub>t+1</sub>, a;&theta;<sub>t</sub>). Because the same parametric model
      is used to both select and to evaluate an action, the Double DQN authors show that over-estimation of value estimates can occur.
      
      The double DQN Algorithm was shown to significantly reduce over-estimation bias in action-value estimates by using the target network
      (parametrized by &theta;<sup>-</sup>) to evaluate the action, while using the online network (parametrized by &theta;) for the action selection.
      
      Y<sub>t</sub><sup>Double-DQN</sup> &equiv; R<sub>t</sub> + &gamma;Q(s<sub>t+1</sub>, argmax<sub>a</sub> Q(s<sub>t+1</sub>, a<sub></sub>; &theta;<sub>t</sub>) &theta;<sub>t</sub><sup>-</sup>)
      
      See the `compute_errors` method of the [Base Policy](agents/policies/base.py) class for code implementation
    
    ###### [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952)
      Rather than performing learning updates on experiences as they are sampled from the environment (i.e. sequentially through time), the DQN
      algorithm uses an experience buffer to store experiences, which are then sampled in a probabilistic way to break correlations between sequential 
      experiences. In this implementation, experiences are stored as tuples of (s<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>, s<sub>t+1</sub>)
      corresponding to the state, action, reward and next_state, respectively. </br> 
      
      During learning, experiences tuples (s<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>, s<sub>t+1</sub>) are drawn non-uniformly from the replay buffer
      according according to the error between the predicted and expected action-value for the given
      experience. Experience samples which result in greater error are given a higher probability for being sampled.
      
      Samples are added to the replay buffer such that the probability of being sampled is:
      
      P(i) = p<sub>i</sub><sup>&alpha;</sup> / &Sum;<sub>k</sub> p<sub>k</sub><sup>&alpha;</sub>
      
      Where p<sub>i</sub> > 0 is the priority of sample i, and &alpha; is used to control how much the priotiziation is used, with
      &alpha;=0 representing the case of uniform sampling. 
      
      To determine the priority of an experience tuple, `TD Error` is used (and can be applied to either the DQN or Double-DQN case). We show
      the case for the Double-DQN, since that is what's implemented in the code:
      
      TD Error = R<sub>t</sub> + &gamma;Q(s<sub>t+1</sub>, argmax<sub>a</sub> Q(s<sub>t+1</sub>, a<sub></sub>; &theta;<sub>t</sub>) &theta;<sub>t</sub><sup>-</sup>) - Q(s<sub>t</sub>, a<sub>t</sub>; &theta;)
      
      In the code, we use the L2 norm of the TD error as our un-normalized priority values p<sub>i</sub>.
      
      Due to the introduced bias from non-uniform sampling of experiences, the gradients are weighted by importance-sampling
      weights as a correction:
      
      w<sub>i</sub> = (1/N * 1/P(i)) <sup>&beta;</sup>
      
      A SumTree data structure is implemented to perform weighted sampling efficiently. See the implementation
      of the [PER buffer](agents/memory/prioritized_memory.py), and the [SumTree](tools/data_structures/sumtree.py).
      
      See the `compute_errors` method of the [Base Policy](agents/policies/base.py) class shows where importance weights
      are applied to scale the gradients, and `step` method of the [DQNAgent](agents/dqn_agent.py) contains the implementation
      of updating the priorities.
      
   ###### [Dueling DQN network](https://arxiv.org/abs/1511.06581)
    The dueling DQN network architecture attempts to decompose Q(s, a) as the sum of the `value`, V(s) of a state, 
    and the `advantage`, A(s, a) of taking an action in that state (advantage over all other possible actions from that state).
    
    Q(s, a) = A(s, a) + V(s)
    
    To accomplish this, the output-component of the DQN is separated into two steams, one for V(s) and one for A(s, a). 
    These two streams are then aggregated to obtain the final Q(s, a) values. 
    
    By decoupling, the Dueling DQN learns the value of a state independently from the value of
    actions from that state, which is particularly useful if actions to not affect the value of a
    state in a relevant way.
    
    For implementation details, see the `_get_dueling_output` method of the [dqn](agents/models/dqn.py).
    
   ###### [Noisy DQN network](https://arxiv.org/abs/1706.10295)
   The Noisy DQN output replaces the traditional &epsilon;-greedy exploration technique for a neural-network layer whose
   weights and biases are perturbed by a parametric function which is learned with the rest of the network. Noisy networks
   have the advantage over  &epsilon;-greedy in that the amount of noise they inject into the network is learned rather than
   annealed in a heuristic manner.
   
   For implementation details, see the `get_output` method of the [dqn](agents/models/dqn.py).

   ###### [Distributional (Categorical) DQN network](https://arxiv.org/abs/1707.06887)
   The categorical DQN algorithm attempts to model the `return distribution` for an action, rather than the
   `expected return`, thus modelling the distribution of Q(s, a). The categorical DQN is implemented in 
   the `get_output` method of [dqn](agents/models/dqn.py), with corresponding [categorical policy](agents/policies/categorical.py)
   which is responsible for computing the errors between the target and online network distributions. Please refer
   to the paper for theoretical details and to this [reference implementation](https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb),
   from which the code is adapted from.

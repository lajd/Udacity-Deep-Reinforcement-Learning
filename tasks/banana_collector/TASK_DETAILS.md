[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Yellow Banana Collector

### Introduction

![Trained Agent][image1]

In this task, the agent's goal is to collect as many yellow bananas as possible while avoiding blue bananas. </br>
There are two variations to this task -- the `Ray-Tracing Banana Collector`, where the state from the environment
is a small (37) dimensional vector of hand-crafted features of the agent's velocity, along with ray-based perception
of objects around agent's forward direction. In contrast, the `Pixel-Based Banana Collector` receives a raw pixel (RGB image)
as state from the environment. Below we go into more details on the task description: </br>

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.
### Action Space
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


### Preparing the Unity ML-Agent environments

If you're running on linux and wish to download both environments, you can run the script [setup_linux.sh](setup_linux.sh).
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

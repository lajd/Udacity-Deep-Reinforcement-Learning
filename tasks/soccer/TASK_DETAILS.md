### Multi Agent Soccer Environment
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

![Soccer][image2]

- Brains:
    - Striker
        - Objective: Get the ball into the Goalie's goal
    - Goalie
        - Objective: Prevent the ball from going into the goal

Agents: The environment contains four agents, two Strikers and two Goalies, where there is a single striker/goalier
pair per team.

Behavior Parameters : Striker, Goalie.
Striker Agent Reward Function (dependent):

+1 When ball enters opponent's goal.
-0.001 Existential penalty.
Goalie Agent Reward Function (dependent):
-1 When ball enters goal.
0.001 Existential bonus.

GoalieBrain actions:
0: forward
1: backward
2: slide right
3: slide left

StrikerBrain actions:
0: forward
1: backward
2: spin right (clockwise)
3: spin left (counter-clockwise)
4: slide left
5: slide right


In this environment, the goal is to train a team of agents to play soccer.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

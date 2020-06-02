[image1]: solution_checkpoint/visual_agent_banana_solution.png "Episode Scores"
[image2]: ../../resources/pixel_banana_collector/rgb_image.png "RGB Image"
[image3]: ../../resources/pixel_banana_collector/gray_image.png "relative lumincance"
[image4]: ../../resources/pixel_banana_collector/hue_image.png "Hue channel"
[image5]: ../../resources/pixel_banana_collector/value_image.png "Value channel"

# Pixel Banana Collector

Please see the [repository overview](../../../../README.md) before reading this report.

## Method
The (hyperparameter_tuning.py)[hyperparameter_tuning.py] module was used to identify candidates with promising performance.
We restricted ourselves to 3-layer CNNs with batch normalization (which was found to be effective) at increasing the 
speed/stability of training. Due to a memory issue with running the Banana.x86_64 environment for more than ~800 episodes, 
along with the slower training speeds, we were not able to perform many experiments.

## Solution model architecture
The best model identified is shown below </br>
```
Convolution output size: 1152
VisualDQN(
  (features): CNN(
    (features): Featurizer(
      (model): Sequential(
        (0): Conv3d(3, 64, kernel_size=(1, 8, 8), stride=(1, 4, 4))
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv3d(64, 128, kernel_size=(1, 4, 4), stride=(1, 2, 2))
        (4): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv3d(128, 128, kernel_size=(4, 3, 3), stride=(1, 3, 3))
        (7): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): Flatten()
      )
    )
  )
  (output): OutputLayer(
    (advantage): NoisyMLP(
      (model): Sequential(
        (0): NoisyLinear()
        (1): ReLU()
        (2): Dropout(p=0.1, inplace=False)
        (3): NoisyLinear()
      )
    )
    (value): NoisyMLP(
      (model): Sequential(
        (0): NoisyLinear()
        (1): ReLU()
        (2): Dropout(p=0.1, inplace=False)
        (3): NoisyLinear()
      )
    )
  )
)

```
which is uses the DQN network with the following extensions: </br>
- `Featurizer`
    - 3-layer 3D CNN with batch normalization and relu activation
    - Kernel sizes of: (1, 8, 8), (1, 4, 4), (4, 3, 3)
    - Stride sizes of: (1, 4, 4), (1, 2, 2), (1, 3, 3)
    - Filters of quantity (64, 128, 128)
- `Noisy outputs`
    - Hidden layer sizes given by the `OUTPUT_FC_HIDDEN_SIZES` parameter
    - A single hidden layer of size (1024, ) was used, with dropout of value 0.1 between hidden layers.
    - ReLu nonlinearity was used
- `Dueling networks`
    - Dueling networks were utilized.


The selected model configuration is as follows: </br>

```
"INITIAL_LR": 5e-4,
"NUM_STACKED_FRAMES": 4,
"OUTPUT_HIDDEN_DROPOUT": 0.1,
"CATEGORICAL": False,
"DUELING": True,
"NOISY": True,
"BATCH_SIZE": 64,
"GRAYSCALE": False,
"N_FILTERS": (64, 128, 128),
"KERNEL_SIZES": [(1, 8, 8), (1, 4, 4), (4, 3, 3)],
"STRIDE_SIZES": [(1, 4, 4), (1, 2, 2), (1, 3, 3)],
"OUTPUT_FC_HIDDEN_SIZES": (1024,),
"WARMUP_STEPS": 5000
```

### Results: 
Below we show the scores for each episode (blue), and the average score over the last 100 episodes (red), until our agent 
has solved the task (average score of 13 over the past 100 episodes) for the architecture and hyper-parameters listed
above. 

The agent solves the task in 656 episodes in 63 minutes.

![Trained Agent][image1]

### Discussion: 
The model architecture was designed similarly to the that of the original DQN paper, however implementing the original
DQN algorithm was shown to result in poor performance (at least in the 800 episode constraint the memory issue
places me under). Increasing the number of filters, and adding batch normalization significantly improved performance, as 
did the addition of the dueling network. In contrast to the Ray-tracing banana collector, at least as a result of
preliminary experiments, the noisy output layer seemed to benefit performance. 

In the DQN paper, the option to reduce the dimensionality of the images by a relative luminance (grayscale) transformation 
was used to significantly improve their performance, however applying the grayscale flag in my experiments showed the opposite, 
likely because colour information was essential to out task of differentiating yellow and blue bananas. 

The images with various dimensionality methods applied are shown below:

Original RGB state: </br>

![RGB_State][image2]

Gray (relative luminance transformation): </br>

![Gray_State][image3]

Hue channel: </br>

![hue][image4]

Value channel: </br>

![value][image5]

Basic experiments were performed with the above dimensionality techniques, which can be found in [tools](../../../../tools/image_utils.py), 
however the network has signfiicant difficulty learning from them (at least in the constraints imposed by the memory issue).

import torch
from torch import nn
from collections import deque


class DQNCNN(nn.Module):
    """ Implements the CNN from the DQN DeepMind paper
    LAYER	KERNEL-SIZE	STRIDE	HIDDEN NODES	ACTIVATION FUNCTION
    CNN1	      8	       4	     NA	                ReLU
    CNN2	      4	       2	     NA	                ReLU
    CNN3	      3	       1	     NA	                ReLU
    FC1	      NA	   NA	     512	            ReLU
    """

    def __init__(self,  action_size: int, num_stacked_frames: int = 4):
        super().__init__()
        self.num_stacked_frames = num_stacked_frames
        self.action_size = action_size

        self.state_buffer = deque(maxlen=num_stacked_frames)
        self.next_state_buffer = deque(maxlen=num_stacked_frames)

        self.model = self.construct_model()

    def construct_model(self):
        cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.num_stacked_frames, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # FC Input size is 1x64x7x7 = 3136
            nn.Linear(3136, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        )
        return cnn

    def forward(self, inp: torch.Tensor):
        return self.model.forward(inp)

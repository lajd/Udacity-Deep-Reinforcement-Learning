import torch
from torch import nn
from collections import deque
from typing import Optional
from agents.models.base import BaseModel
from agents.models.components.mlp import MLP
from agents.models.components.noisy_mlp import NoisyMLP
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN(BaseModel):
    """ DQN network

    Implements various progressions of the DQN algorithm
    [https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf]
    towards the Rainbow DQN algorithm [https://arxiv.org/pdf/1710.02298.pdf]

    The repository https://github.com/higgsfield/RL-Adventure/ was used for inspiration

    Toggling extensions can be done through input arguments.

    Extensions include:
        - Double DQN
        - Dueling DQN
        - Categorical DQN
    """

    def __init__(
            self,
            state_shape: tuple,
            action_size: int,
            featurizer: torch.nn.Module,
            feature_size: int,
            seed: Optional[int] = None,
            grayscale: bool = False,
            num_stacked_frames: int = 4,
            output_hidden_layer_size: tuple = (512,),
            output_hidden_dropout: Optional[float] = None,
            dueling_output: bool = True,
            noisy_output: bool = True,
            # Only used when categorical_output=True
            categorical_output: bool = True,
            categorical_num_atoms: int = 51,
            categorical_v_min: int = -10,
            categorical_v_max: int = 10,
            **kwargs
    ):
        """
        Args:
            state_shape: The shape of the states to be fed through. Used for initialization
            action_size: The size of the action space
            featurizer (torch.nn.Module): A component for taking input observations to a flattened feature vectors (batch_size, -1)
            feature_size: (int) The flattened size of the feature vector obtained from featurizer.forward()
            seed: Optional[int] = None: The random seed to initialize module children with
            grayscale: bool = Whether to convert the input RGB (3 channel )images to grayscale (single channel),
            num_stacked_frames: int = 4: The number of frames to stack for the input volume. For an image of n_image_channels,
                this created an input volume of shape (b_size, n_image_channels, num_stacked_frames, W, H) during learning
            output_hidden_layer_size: tuple = (512,): Hidden layer sizes taking the feature vector to Q(s, a)
            output_hidden_dropout: Optional[float] = None: Dropout between layers of the hidden layers
            dueling_output: bool = True: Flag dueling DQN
            noisy_output: bool = True: Flag noisy DQN
            categorical_output: bool = True, Flag categorical DQN
            categorical_num_atoms: int = 51 : Generate distributions of Q(s, a) of shape (batch_size, -1)
            categorical_v_min: int = -10: Minimum support in categorical DQN
            categorical_v_max: int = 10: Maximum support in categorical DQN
        """
        super().__init__()

        if seed:
            self.set_seed(seed)

        # General params
        self.seed = seed
        self.num_stacked_frames = num_stacked_frames
        self.grayscale = grayscale
        self.state_shape = state_shape
        self.action_size = action_size
        self.feature_size = feature_size

        # Model head hyper-params
        self.output_hidden_layer_size = output_hidden_layer_size
        self.output_hidden_dropout = output_hidden_dropout

        # dueling_output DQN
        self.dueling_output = dueling_output

        # noisy_output DQN
        self.noisy_output = noisy_output

        # categorical_output DQN
        self.categorical_output = categorical_output
        self.categorical_v_min = categorical_v_min
        self.categorical_v_max = categorical_v_max
        self.categorical_num_atoms = categorical_num_atoms
        self.categorical_support = torch.linspace(self.categorical_v_min, self.categorical_v_max, self.categorical_num_atoms).to(device)

        # State buffer
        self.state_buffer = deque(maxlen=self.num_stacked_frames)

        # Child modules for obtaining features and output
        self.features = featurizer
        self.output = self.get_output()

    def step(self):
        """Perform actions after each learning step"""
        if self.noisy_output:
            self.output.reset_noise()

    def step_episode(self, episode: int):
        """Perform actions after each episode"""
        self.state_buffer.clear()

    def get_output(self):
        """ Get the output layer for the forward pass for the flavours of DQN

        Handles output layer for: Dueling DQN, Categorical DQN and Noisy DQN
        """
        layer_sizes = [self.feature_size] + list(self.output_hidden_layer_size)

        if self.noisy_output:
            output_type = NoisyMLP
        else:
            output_type = MLP

        if self.dueling_output:
            if self.categorical_output:
                value_layer_sizes = layer_sizes + [self.categorical_num_atoms]
                advantage_layer_sizes = layer_sizes + [self.action_size * self.categorical_num_atoms]
            else:
                value_layer_sizes = layer_sizes + [1]
                advantage_layer_sizes = layer_sizes + [self.action_size]

            value = output_type(layer_sizes=tuple(value_layer_sizes), dropout=self.output_hidden_dropout)
            advantage = output_type(layer_sizes=tuple(advantage_layer_sizes), dropout=self.output_hidden_dropout)
            return self._get_dueling_output(advantage, value, self.categorical_output, self.categorical_num_atoms, self.action_size)

        else:
            if self.categorical_output:
                layer_sizes += [self.action_size * self.categorical_num_atoms]
            else:
                layer_sizes += [self.action_size]

            linear_output = output_type(tuple(layer_sizes), dropout=self.output_hidden_dropout)
            return self._get_output(linear_output)

    def _get_output(self, linear_output: torch.nn.Module):
        """ Get output layer for a non-dueling architecture"""
        noisy_output = self.noisy_output
        categorical_output = self.categorical_output
        num_categorical_atoms = self.categorical_num_atoms
        action_size = self.action_size
        class OutputLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.out = linear_output
                self.noisy = noisy_output
                self.categorical = categorical_output

            def forward(self, x: torch.Tensor):
                batch_size = x.shape[0]
                if self.categorical:
                    q_atoms = self.out(x).view(batch_size, action_size, num_categorical_atoms)
                    return q_atoms
                else:
                    q = self.out(x)
                    return q

            def reset_noise(self):
                if self.noisy:
                    self.out.reset_noise()

        return OutputLayer()

    def _get_dueling_output(self, advantage: torch.nn.Module, value: torch.nn.Module, categorical_output: bool, categorical_num_atoms: int, action_size):
        """ Get output for a dueling network architecture """
        noisy_output = self.noisy_output

        class OutputLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.advantage = advantage
                self.value = value
                self.noisy = noisy_output
                self.categorical = categorical_output
                self.num_categoical_atoms = categorical_num_atoms

            def forward(self, x: torch.Tensor):
                batch_size = x.shape[0]
                value_ = self.value(x)
                advantage_ = self.advantage(x)
                if self.categorical:
                    value_ = value_.view(batch_size, 1, categorical_num_atoms)
                    advantage_ = advantage_.view(batch_size, action_size, categorical_num_atoms)
                    q_atoms = value_ + advantage_ - advantage_.mean(dim=1, keepdim=True)
                    return q_atoms
                else:
                    q = value_ + advantage_ - advantage_.mean()
                    return q

            def reset_noise(self):
                if self.noisy:
                    self.advantage.reset_noise()
                    self.value.reset_noise()

        return OutputLayer()

    def forward(self, state, act=False) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        if self.categorical_output:
            dist = self.dist(state, act)
            q = torch.sum(dist * self.categorical_support, dim=2)
            return q
        else:

            state = self.prepare_for_forward(state, act)
            features = self.features(state)

            q = self.output(features)
            return q

    def dist(self, x: torch.Tensor, act=False) -> torch.Tensor:
        """ Obtain the categorical distribution over Q(a, a)"""
        if not self.categorical_output:
            raise ValueError("Dist is only applicable when using categorical_output DQN")
        x = self.prepare_for_forward(x, act=act)
        feature = self.features(x)  # (batch, -1)
        q_atoms = self.output(feature)  # (batch, )
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        return dist

    def prepare_for_forward(self, state, act=False):
        """Perform preparation of the state each time the state is passed to the agent"""
        return state

    def preprocess_state(self, state: torch.Tensor):
        """ Perform 1-time preprocessing of the state before it reaches the agent"""
        return state


class VisualDQN(DQN):
    def __init__(
            self,
            state_shape,
            action_size,
            featurizer: torch.nn.Module,
            feature_size: int,
            seed: Optional[int] = None,
            grayscale: bool = False,
            num_stacked_frames: int = 4,
            output_hidden_layer_size: tuple = (512,),
            output_hidden_dropout: Optional[float] = None,
            dueling_output: bool = True,
            noisy_output: bool = True,
            categorical_output: bool = True,
            categorical_num_atoms: int = 51,
            categorical_v_min: int = -10,
            categorical_v_max: int = 10,
            **kwargs):
        super().__init__(
            state_shape=state_shape,
            action_size=action_size,
            featurizer=featurizer,
            feature_size=feature_size,
            seed=seed,
            grayscale=grayscale,
            dueling_output=dueling_output,
            num_stacked_frames=num_stacked_frames,
            output_hidden_layer_size=output_hidden_layer_size,
            output_hidden_dropout=output_hidden_dropout,
            noisy_output=noisy_output,
            categorical_output=categorical_output,
            categorical_num_atoms=categorical_num_atoms,
            categorical_v_min=categorical_v_min,
            categorical_v_max=categorical_v_max,
        )

    def prepare_for_forward(self, state: torch.FloatTensor, act: bool = False):
        """Build a network that maps state -> action values.

        act: Whether to expect a single sample (i.e. not a training batch) and to supplement
        frames from the state buffer
        """
        if act:
            self.state_buffer.append(state)
            # Ensure the state buffer has at least num_stacked_frames states
            while len(self.state_buffer) < self.num_stacked_frames:
                self.state_buffer.appendleft(self.state_buffer[0])
            # Stack over the frames dimension
            state = torch.cat(list(self.state_buffer), dim=0)
            # Add the batch dimension
            state = state.unsqueeze(0)

        if not self.grayscale:
            # Reshape as batch x channels x depth x width x height for pytorch CNN
            state = state.permute(0, 4, 1, 2, 3)

        return state

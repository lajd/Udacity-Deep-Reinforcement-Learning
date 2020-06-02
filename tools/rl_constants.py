import torch
from collections import namedtuple
from typing import List, Union

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "t_step"])

Environment = namedtuple("Environment", field_names=["next_state", "reward", "done"])

Action = namedtuple('Action', field_names=["value", "distribution"])

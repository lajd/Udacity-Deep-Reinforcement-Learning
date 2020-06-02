import torch
from collections import defaultdict
from typing import List


class ParameterCapture:
    def __init__(self, add_mod: int = 100, max_size: int = 10000):
        self.parameters = defaultdict(lambda: {'count': 0, 'values': []})
        self.add_mod = add_mod
        self.max_size = max_size

    def downsample(self, param_list: List[torch.Tensor]):
        new_list = []
        for i, p in enumerate(param_list):
            if i % 2 == 0:
                new_list.append(p)
        return new_list

    def add(self, k: str, value: torch.Tensor):
        value = value.cpu().data.numpy()
        if self.parameters[k]['count'] % self.add_mod == 0:
            self.parameters[k]['values'].append(value)
        if self.parameters[k]['count'] > self.max_size:
            self.parameters[k]['values'] = self.downsample(self.parameters[k]['values'])

    def get(self, k: str):
        return self.parameters[k]['values']
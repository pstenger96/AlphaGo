import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np

LAYER_SIZE = 256
class Net():
    def __init__(self, input_size, n_actions, type, device):
        super(Net, self).__init__()

        self.model = 0
        if type == "DNN":
            self.model = DNN(input_size, n_actions)
        elif type=="CNN_MCTS":
            self.model = CNN_MCTS(input_size, n_actions)
        elif type=="CNN_DQN":
            self.model = CNN_DQN(input_size, n_actions)
        self.model.to(device)




class NN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(NN, self).__init__()

        self.policy = 0
        self.value = 0


    def forward(self, x):
        return 0

class DNN(NN):
    def __init__(self, input_size, n_actions):
        super(DNN, self).__init__(input_size, n_actions)

        self.policy = nn.Sequential(
            nn.Linear(input_size[0], LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER_SIZE, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(input_size[0], LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER_SIZE, 1)
        )

    def forward(self, x):
        return self.policy(x.float()), self.value(x.float())

class CNN_MCTS(NN):
    def __init__(self, input_size, n_actions):
        super(CNN_MCTS, self).__init__(input_size, n_actions)

        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_size)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.policy(conv_out.float()), self.value(conv_out.float())


class CNN_DQN(NN):

    def __init__(self, input_size, n_actions):
        super(CNN_DQN, self).__init__(input_size, n_actions)

        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_size)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


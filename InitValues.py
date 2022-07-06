from abc import ABC, abstractmethod
import numpy as np
import torch
class InitValues(ABC):
    @abstractmethod
    def getInitVal(self, state):
        pass


class SimpleInit(InitValues):
    def __init__(self, amount_actions):
        super().__init__()
        self.amount_actions = amount_actions

    def getInitVal(self, state):
        return [0.0] * self.amount_actions


class PretrainedInit(InitValues):
    def __init__(self, net, tau, device):
        super().__init__()
        self.net = net
        self.tau = tau
        self.device = device

    def getInitVal(self, state):
        state = torch.FloatTensor(state).unsqueeze(dim = 0).to(self.device)
        q_vals = self.net.model(state).cpu().data.numpy()[0]
        tau_div = q_vals/self.tau
        exp = np.exp(tau_div)
        sum_ = np.log(sum(exp)) * self.tau
        val =  (q_vals - sum_) / self.tau
        for v in val:
            if v > 0:
                dbg = 0
        return val
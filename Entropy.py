from abc import ABC, abstractmethod
import numpy as np
import torch

EPSILON = 0.000001

class Entropy(ABC):
    def __init__(self, q_vals):
        self.Q = q_vals
        self.maxQ = -np.inf

    @abstractmethod
    def Backup(self, next_state_idx, tau):
        pass
    @abstractmethod
    def Policy(self, current_state_idx, tau):
        pass

    def SetMaxQ(self):
        self.maxQ = -np.inf
        for key in self.Q:
            loc_max = max(self.Q[key])
            if loc_max > self.maxQ:
                self.maxQ = loc_max

class MENTS_Entropy(Entropy):
    def __init__(self, q_vals):
        super().__init__(q_vals)

    def Backup(self, next_state_idx, tau):
        self.SetMaxQ()
        value_softmax_next = self.Q[next_state_idx]
        val = max(np.sum(np.exp((np.array(value_softmax_next) - self.maxQ) / tau)), EPSILON)
        return tau * np.log(val) + self.maxQ

    def Policy(self, current_state_idx, tau):
        self.SetMaxQ()
        values_sftmx = self.Q[current_state_idx]
        sum_q = np.sum(np.exp((np.array(values_sftmx) - self.maxQ) / tau))
        q_div = np.exp((np.array(values_sftmx) - self.maxQ) / tau)
        return q_div / sum_q



class TENTS_Entropy(Entropy):
    def __init__(self, q_vals):
        super().__init__(q_vals)
        self.K = []
    def Backup(self, next_state_idx, tau):
        value_softmax_next = np.array(self.Q[next_state_idx]) / tau
        self.GetK(value_softmax_next)
        left_summand = 0.5*np.multiply(value_softmax_next, value_softmax_next)
        right_s_filtered = np.sum(value_softmax_next[self.K])

        right_summand = ((right_s_filtered - 1) * (right_s_filtered -1)) / (2*len(self.K)*len(self.K))
        spmax = np.sum((left_summand - right_summand)[self.K]) + 0.5
        return tau* spmax

    def Policy(self, current_state_idx, tau):
        value_softmax = np.array(self.Q[current_state_idx])
        self.GetK(value_softmax / tau)
        left_summand = value_softmax / tau
        right_summand = np.sum(((value_softmax / tau)  - 1)[self.K]) / len(self.K)
        pol = []
        for p in left_summand:
            pol.append(max(p - right_summand, 0))
        return np.array(pol)

    def GetK(self, value):
        vals_np = np.array(value)
        sorted_vals = np.zeros((vals_np.shape[0], 2))
        for k in range(len(vals_np)):
            sorted_vals[k, 0] = vals_np[k]
            sorted_vals[k, 1] = k
        tmp = - sorted_vals
        sorted_vals = sorted_vals[tmp[:, 0].argsort()]
        self.K = []
        #self.K.append(int(sorted_vals[0, 1]))

        for i in range(len(value)):
            indices = np.arange(0, i+1)
            sum = sorted_vals[indices, 0].sum()
            if 1 + (i + 1) * sorted_vals[i, 0] > sum:

                self.K.append(int(sorted_vals[i, 1]))
        if len(self.K) == 0:
            dng = 0

        self.K = np.array(self.K)


class RENTS_Entropy(Entropy):
    def __init__(self, q_vals, epsilon, amount_actions, counts, discStates, prev_net, idx_to_state):
        super().__init__(q_vals)
        self.prev_pol = {}
        self.epsilon = epsilon
        self.amount_actions = amount_actions
        self.counts = counts
        self.idx_list = []

        pol_list = []
        for _ in range(amount_actions):
            pol_list.append(1.0 / amount_actions)
        for s in discStates:
            s_arr = np.array(s)
            state_idx = s_arr.data.tobytes()
            self.idx_list.append(state_idx)
            self.prev_pol[state_idx] = pol_list
        self.prev_net = prev_net
        self.idx_to_state = idx_to_state
        dbg = 0

    def Backup(self, next_state_idx, tau):
        self.SetMaxQ()

        prev_p, _ = self.prev_net(torch.tensor(self.idx_to_state[next_state_idx]).to("cuda"))
       # prev_p = list(self.prev_pol[next_state_idx])
        value_softmax_next = self.Q[next_state_idx]
        product = np.multiply(np.array(prev_p.data.cpu()), np.exp((np.array(value_softmax_next) - self.maxQ) / tau))

        return tau * np.log(np.sum(product)) + self.maxQ

    def Policy(self, current_state_idx, tau):

        self.SetMaxQ()

        prev_pol, _ = self.prev_net(torch.tensor(self.idx_to_state[current_state_idx]).to("cuda"))
       # prev_pol = self.prev_pol[current_state_idx]

        values_sftmx = self.Q[current_state_idx]
        sum_q = np.sum( np.multiply(prev_pol.data.cpu().numpy(),  np.exp((np.array(values_sftmx) - self.maxQ) / tau)))
        q_div = prev_pol.data.cpu().numpy() * np.exp((np.array(values_sftmx) - self.maxQ) / tau)

       # self.CalcPrevPolicy(tau)
        return q_div / sum_q

    def CalcPrevPolicy(self, tau):

        for idx in self.Q.keys():
         #   if idx in self.Q:
            val_softmax = np.array(self.Q[idx])

            #else:
             #   val_softmax = np.zeros((self.amount_actions, 1))
          #  if idx in self.counts:
            counts = self.counts[idx]
         #   else:
           #     counts = np.zeros((self.amount_actions, 1))
            self.maxQ = 0


            sum_q = np.sum(np.multiply(self.prev_pol[idx], np.exp((val_softmax - self.maxQ) / tau)))
            q_div = self.prev_pol[idx] * np.exp((val_softmax - self.maxQ) / tau)



            lambda_s = self.epsilon * self.amount_actions / np.log(sum(counts) + 1)

            if lambda_s == np.inf or lambda_s == np.nan:
                pol = []
                for _ in range(self.amount_actions):
                    pol.append(1.0 / self.amount_actions)
                pol = np.array(pol)
            else:
                pol = (1 - lambda_s) * (q_div / sum_q) + lambda_s / self.amount_actions
                sum_p = sum(pol)
                for i in range(len(pol)):
                    pol[i] = pol[i] / sum_p

            self.prev_pol[idx] = pol

        dbg = 0

class AlphaDiv(Entropy):
    def __init__(self, q_vals, alpha):
        super().__init__(q_vals)
        self.alpha = alpha
        self.current_pol = 0

    def Backup(self, next_state_idx, tau):
        f_alpha = ((np.power(self.current_pol, self.alpha) - 1) - self.alpha*(self.current_pol -1))/(self.alpha*(self.alpha - 1))
        bell_Q = 0
        return max(bell_Q - tau * f_alpha)

    def Policy(self, current_state_idx, tau):
        pass

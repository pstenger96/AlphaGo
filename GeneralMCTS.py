import math
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from Entropy import Entropy, MENTS_Entropy, TENTS_Entropy, RENTS_Entropy
from Discretization import Discretization, CartpoleDiscretization, AcrobotDiscretization, NoDiscretization
from InitValues import PretrainedInit, SimpleInit
class GeneralMCTS:
    def __init__(self, entr_string, init_string, env_name, prev_net, device, pretrained_net = None):
        self.visit_counts = {}

        self.values_softmax = {}
        self.probs = {}
        self.idx_to_state = {}

        self.tau = 0.8
        self.epsilon = 0.25

        self.amount_actions = 6

        self.acc_reward = 0.0
        self.dc = 0.99
        self.step =  0
        self.totalMaxAbsoluteReward = 100.0

        self.discretization = 0
        if env_name == "CartPole-v0":
            self.discretization = CartpoleDiscretization()
        elif env_name == "Acrobot-v1":
            self.discretization = AcrobotDiscretization()
        else:
            self.discretization = NoDiscretization()

        disStates = self.discretization.getDiscretization()
        """
        for s in disStates:
            s_arr = np.array(s)
            state_idx = s_arr.data.tobytes()
            self.idx_to_state[state_idx] = s
        """

        if entr_string == "MENTS":
            self.Entropy_ = MENTS_Entropy(self.values_softmax)
        elif entr_string == "TENTS":
            self.Entropy_ = TENTS_Entropy(self.values_softmax)
        elif entr_string == "RENTS":
            self.Entropy_ = RENTS_Entropy(self.values_softmax,  self.epsilon, self.amount_actions, self.visit_counts, disStates, prev_net, self.idx_to_state)

        if init_string == "Simple":
            self.Init = SimpleInit(self.amount_actions)
        elif init_string == "Pretrained":
            self.Init = PretrainedInit(pretrained_net, self.tau, device)




    def __len__(self):
        return len(self.values_softmax)


    def clear(self):
        self.visit_counts.clear()

        self.values_softmax.clear()
        self.probs.clear()

    def getStateIndex(self, state):
        closest = self.discretization.getClosestState(state)
        return closest, closest.data.tobytes()


    def find_leaf(self, env, state_int):
        discr_state, state_idx = self.getStateIndex(state_int)
        self.idx_to_state[state_idx] = discr_state
        states = []
        state_idx_dbg = []
        actions = []
        cur_state = state_int
        self.acc_reward = 0
        self.step = 0
        values = []
        while not self.is_leaf(state_idx):


            states.append(state_idx)
            #state_idx_dbg.append(self.getIdxInDiscr(state_idx))
            counts = self.visit_counts[state_idx]

            if sum(counts) > 0:

                lambda_s = self.epsilon * self.amount_actions / np.log(sum(counts) + 1)
                pol = (1 - lambda_s) *  self.Entropy_.Policy(state_idx, self.tau) + lambda_s / self.amount_actions
                for p in pol:
                    if np.isnan(p) or p < 0 or sum(pol) > 1.01001 or sum(pol) < 0.9900:
                        dbg = 0
                sum_pol = sum(pol)
                if pol[0] != pol[1]:
                    dbg = 0
                for pol_idx in range(len(pol)):
                    pol[pol_idx] = pol[pol_idx] / sum_pol


                action = np.random.choice(self.amount_actions, p = pol)
            else:
                action = np.random.choice(self.amount_actions)

            actions.append(action)
            cur_state, r, done, _ = env.step(action)

           # r = r + 1

            values.append(r)
            if done:
                break

            discr_state, state_idx = self.getStateIndex(cur_state)
            self.idx_to_state[state_idx] = discr_state
            self.step+=1

        return values, cur_state, states,actions, env

    def is_leaf(self , state_init_idx):
        return state_init_idx not in self.probs




    def getLeafValidStates(self, env):
        env_list = []
        for _ in range(self.amount_actions):
            env_list.append(deepcopy(env))
        states = []
        for i in range(len(env_list)):
            state, r, done, _ = env_list[i].step(i)
            states.append(self.discretization.getClosestState(state))
        for j in range(len(env_list)):
            env_list[j].close()
        return np.array(states)


    def search_batch(self, env, count, batch_size, state_init, net, device="cpu"):
        for i in range(count):
            self.search_minibatch(env, batch_size, state_init, net, device)

        env.close()

    def search_minibatch(self, env, count, state_init, net, device="cpu"):
        backup_queue = []
        expand_states = []

        expand_queue = []
        planned = set()

        actions = []
        states_idx_list = []
        for _ in range(count):
            values, leaf_state, states_idx_list, actions, current_env = self.find_leaf(deepcopy(env), state_init)

            if len(values) > 0:
                backup_queue.append((values, states_idx_list, actions))
            else:
                _, leaf_state_idx =  self.getStateIndex(leaf_state)
                if leaf_state_idx not in planned:
                    planned.add(leaf_state_idx)
                    leaf_state_lists = self.getLeafValidStates(current_env)
                    expand_states.append(leaf_state_lists)

                    expand_queue.append((leaf_state_idx, states_idx_list, actions))
            current_env.close()
        if expand_queue:
            batch_v = torch.tensor(expand_states[0]).to(device)
            policy, values_v = net(batch_v)
            probs_v = F.softmax(policy, dim=1)
            values_net = values_v.data.cpu().numpy()[:, 0]
            probs = probs_v.data.cpu().numpy().squeeze()

            idc = 0
            for (leaf_state_idx, states_idx_list, actions), value, prob in zip(expand_queue, values_net, probs):
                self.visit_counts[leaf_state_idx] = [0] * self.amount_actions
                self.values_softmax[leaf_state_idx] = self.Init.getInitVal(self.idx_to_state[leaf_state_idx])
                self.probs[leaf_state_idx] = prob

                idc+=1
                backup_queue.append(([value], states_idx_list, actions))




        for values, state_idx, actions in backup_queue:
            idx = len(values) - 1
            if len(values) > 10:
                dbg = 0

            values_v = 0
            if len(values) > 0 and len(state_idx) > 0 :
                self.visit_counts[state_idx[idx]][actions[idx]] += 1
                state = self.idx_to_state[state_idx[idx]]
                policy, values_v = net(torch.FloatTensor(state).unsqueeze(dim= 0).to("cuda"))
                self.values_softmax[state_idx[idx]][actions[idx]] = values[idx] + self.dc * values_v.item()

            idx -= 1
            if len(values) > 1 and len(state_idx) > 1:
                for _ in range(len(state_idx) - 1):
                        self.visit_counts[state_idx[idx]][actions[idx]] +=1
                        next_state_idx = state_idx[idx + 1]
                        self.values_softmax[state_idx[idx]][actions[idx]] = (values[idx] +  self.dc *self.Entropy_.Backup(next_state_idx=next_state_idx, tau=self.tau))

                        idx-=1



    def get_policy_value(self, state_int, tau=1):
        _, state_idx = self.getStateIndex(state_int)
        counts = self.visit_counts[state_idx]
        if tau==0:
            probs = [0.0] * self.amount_actions
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count **(1.0/tau) for count in counts]
            total = sum(counts)
            probs = [count/total for count in counts]

        values = self.values_softmax[state_idx]
        return probs, values
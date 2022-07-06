import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

"""
LAYER_SIZE = 256
class Net(nn.Module):
    def __init__(self, input_size, n_actions):
        super(Net, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_size, LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER_SIZE, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(input_size, LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER_SIZE, 1)
        )

    def forward(self, x):

        return self.policy(x.float()), self.value(x.float())
"""

def play_game(env, mcts, replay_buffer, net, steps_before_tau_0, mcts_searches, mcts_batch_size, device = "cpu"):

    state = env.reset()


    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = 0

    is_done = False

    while not is_done:
        mcts.search_batch(deepcopy(env), mcts_searches, mcts_batch_size, state, net, device=device)
        probs, _ = mcts.get_policy_value(state, tau=tau)
        game_history.append((state, probs, result))
        action = np.random.choice(env.action_space.n, p=probs)
        state, r, is_done, _ = env.step(action)

       # r = r + 1
        result += 0.99 ** step * r
        if is_done:
            break



        step += 1
        print(step)
        if step >= steps_before_tau_0:
            tau = 0

        if step == 2:
            is_done = True
    if replay_buffer is not None:
        for state, probs, result in reversed(game_history):
            replay_buffer.append((state, probs, result))

    return result, step
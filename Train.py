from ModelType import Net, NN, DNN
from Model import play_game
from MCTS import MCTS
from MCTS_MENTS import MCTS_MENTS
from GeneralMCTS import GeneralMCTS
from Entropy import Entropy, MENTS_Entropy
import gym
from Utils import TargetNet, TBMeanTracker, RewardTracker
import torch
import collections
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
import time
from AtariWrapper import make_env, make_env_simple

LEARNING_RATE = 0.1
BUFFER_SIZE = 10000
MIN_REPLAY_TO_TRAIN = 7000
MCTS_SEARCHES = 20
MCTS_BATCH_SIZE = 20
PLAY_EPISODES = 1
STEPS_BEFORE_TAU_0 = 0
TRAIN_ROUNDS = 5
BATCH_SIZE = 32
EVALUATE_EVERY_STEP = 20
EVALUATION_ROUNDS = 20
SAVE_DATA_EVERY_STEP = 10
STEPS = 100
ENV_NAME = "PongNoFrameskip-v4"
ModelType = "CNN_MCTS"       #DNN, CNN_MCTS
Used_Entropy = "MENTS" #MENTS, RENTS, TENTS
InitType = "Pretrained" #Simple, Pretrained

def evaluate(net, prev_net, rounds, device, pretrained_net = None):
    mcts = GeneralMCTS(Used_Entropy, InitType, ENV_NAME, prev_net.target_model, device ,pretrained_net=pretrained_net)
    env = make_env(ENV_NAME)
    avg_rew = 0
    for i in range(rounds):
        print(i)
        reward, _ = play_game(env, mcts, replay_buffer=None, net=net.model, steps_before_tau_0=0, mcts_searches=30, mcts_batch_size=20, device = device)
        print(reward)
        avg_rew+=reward
    return avg_rew/rounds

def train():

    x_data = []
    y_data = []
    file = open("data.txt", 'a')

    seeds = [1, 2, 3, 4]



    for s in range(len(seeds)):
        random.seed(seeds[s])
        np.random.seed(seeds[s])

        current_best = -np.inf
        device = "cuda"
        env = make_env(ENV_NAME)
        net = Net(env.observation_space.shape, env.action_space.n, ModelType, device)
        net.model.eval()
        best_net = TargetNet(net.model)
        prev_net = TargetNet(net.model)

        pretrained_net = None
        if InitType == "Pretrained":
            pretrained_net = Net(env.observation_space.shape, env.action_space.n, "CNN_DQN", device)
            pretrained_net.model.load_state_dict(torch.load('DQN_Model_Data/PongData.dat'))


        optimizer = torch.optim.Adam(net.model.parameters(), lr=LEARNING_RATE)
        replayBuffer = collections.deque(maxlen=BUFFER_SIZE)
        mcts = GeneralMCTS(Used_Entropy, InitType, ENV_NAME, prev_net.target_model, device,  pretrained_net=pretrained_net)
        step_idx = 0
        best_idx = 0
        writer = SummaryWriter()

        """
        score = evaluate(net, EVALUATION_ROUNDS, device)
        x_data.append(step_idx)
        y_data.append(score)
        file.write(str(step_idx))
        file.write(" ")
        file.write(str(score))
        file.write("\n")
        file.close()
        """
        #file = open("data122.txt", 'a')

        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while step_idx < STEPS:
                t = time.time()
                prev_nodes = len(mcts)
                game_steps = 0
                for _ in range(PLAY_EPISODES):
                    rew, steps = play_game(env,mcts, replayBuffer, best_net.target_model, steps_before_tau_0=STEPS_BEFORE_TAU_0, mcts_searches= MCTS_SEARCHES, mcts_batch_size=MCTS_BATCH_SIZE, device = device)
                    file = open("data.txt", 'a')
                    file.write(str(step_idx))
                    file.write(" ")
                    file.write(str(steps))
                    file.write(" ")
                    file.write(str(rew))
                    file.write("\n")
                    file.close()
                    game_steps+= steps

                game_nodes = len(mcts) - prev_nodes
                dt = time.time() - t
                speed_steps = game_steps / dt
                speed_nodes = game_nodes / dt
                tb_tracker.track("speed_steps", speed_steps, step_idx)
                tb_tracker.track("speed_nodes", speed_nodes, step_idx)
                print("Step %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d, replay %d" % (
                    step_idx, game_steps, game_nodes, speed_steps, speed_nodes, best_idx, len(replayBuffer)))

                step_idx+=1
                if step_idx == 40:
                    dbg = 0
                if len(replayBuffer) < MIN_REPLAY_TO_TRAIN:
                    continue

                sum_loss = 0.0
                sum_value_loss = 0.0
                sum_policy_loss = 0.0

                prev_net.sync()
                net.model.train()
                for _ in range(TRAIN_ROUNDS):
                    batch = random.sample(replayBuffer, BATCH_SIZE)
                    batch_states, batch_probs, batch_values = zip(*batch)

                    states_v = torch.FloatTensor(np.array(batch_states)).to(device)
                    optimizer.zero_grad()
                    probs_v = torch.FloatTensor(batch_probs).to(device)
                    values_v = torch.FloatTensor(batch_values).to(device)
                    out_policies, out_values = net.model(states_v)

                    loss_value_v = F.mse_loss(out_values.squeeze(-1), values_v)
                    loss_policy_v = -F.log_softmax(out_policies, dim=1) * probs_v
                    loss_policy_v = loss_policy_v.sum(dim=1).mean()

                    loss_v = loss_policy_v + loss_value_v
                    loss_v.backward()
                    optimizer.step()

                    sum_loss += loss_v.item()
                    sum_value_loss += loss_value_v.item()
                    sum_policy_loss += loss_policy_v.item()

                net.model.eval()
                tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, step_idx)
                tb_tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, step_idx)
                tb_tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, step_idx)


                if step_idx % EVALUATE_EVERY_STEP == 0:
                    score = evaluate(net, prev_net, EVALUATION_ROUNDS, device, pretrained_net)
                    x_data.append(step_idx)
                    y_data.append(score)

                    print("Net evaluated, win ratio = %.2f" % score)
                    writer.add_scalar("eval_win_ratio", score, step_idx)
                    if score > current_best:
                        current_best = score
                        print("Net is better than cur best, sync")
                        best_idx+=1
                        best_net.sync()
                        f = 'run1' + str(best_idx)
                        torch.save(net.model.state_dict(), f)
                        mcts.clear()
        file = open("data.txt", 'a')
        file.write("STOP")
        file.write("\n")
        file.close()
    file.close()


if __name__ == "__main__":
    train()
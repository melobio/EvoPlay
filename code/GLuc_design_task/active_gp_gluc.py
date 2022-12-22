from __future__ import print_function
import argparse
import random
from collections import defaultdict, deque
from sequence_env_gp_gluc import Seq_env, Mutate
from mcts_alphaZero_mutate import MCTSMutater
#from p_v_net_torch import PolicyValueNet  # Pytorch
from p_v_net_2 import PolicyValueNet
#from env_model import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
import time

from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
import os
import pandas as pd


gluc_wt_sequence = "KPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPGFKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIKGAGGD"


def train_regr_predictor(features, fitness, seed):
    X_GP = features
    Y_GP = fitness
    X_GP = np.asarray(X_GP)
    Y_GP = np.asarray(Y_GP)
    regr = GaussianProcessRegressor(random_state=seed)
    regr.fit(X_GP, Y_GP)
    return regr

def run_Clustering(features, n_clusters, subclustering_index=np.zeros([0])):
    if len(subclustering_index) > 0:
        features_sub = features[subclustering_index, :]
    else:
        features_sub = features

    kmeans = KMeans(n_clusters=n_clusters).fit(features_sub)
    cluster_labels = kmeans.labels_

    Length = []
    Index = []

    if len(subclustering_index) > 0:
        for i in range(cluster_labels.max() + 1):
            index = subclustering_index[np.where(cluster_labels == i)[0]]
            l = len(index)
            Index.append(index)
            Length.append(l)
    else:
        for i in range(cluster_labels.max() + 1):
            index = np.where(cluster_labels == i)[0]
            l = len(index)
            Index.append(index)
            Length.append(l)

    return Index


def shuffle_index(Index):
    for i in range(len(Index)):
        np.random.shuffle(Index[i])

    return Index

###
def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:

    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


seed = 100
AAS = "ILVAGMFYWEDQNHCRKSTP"

def feature_single(variant):
    Feature = []
    aalist = list(AAS)
    for AA in variant:
        Feature.append([AA == aa for aa in aalist])
    Feature = np.asarray(Feature).astype(float)
    if len(Feature.shape) == 2:
        features = np.reshape(Feature, [Feature.shape[0] * Feature.shape[1]])

    return features

AAS = "ILVAGMFYWEDQNHCRKSTP"

class TrainPipeline():
    def __init__(self, start_seq_pool, alphabet, model, trust_radius, init_model=None): #init_model=None  feature_list, #, combo_feature_map, combo_index_map, first_round_index, round,
        # params of the board and the game
        self.seq_len = len(start_seq_pool[0])
        self.vocab_size = len(alphabet)
        self.n_in_row = 4

        self.round = round
        #self.combo_index_map = combo_index_map
        # index_combo_map = {v: k for k, v in combo_index_map.items()}
        # first_round_combo = [index_combo_map[ind] for ind in first_round_index]
        self.seq_env = Seq_env(
            self.seq_len,
            alphabet,
            model,
            start_seq_pool,
            #combo_feature_map,
            #first_round_index,
            #combo_index_map,
            #first_round_combo,
            #feature_list,
            trust_radius)  #n_in_row=self.n_in_row
        self.mutate = Mutate(self.seq_env)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 40  # num of simulations for each move 400
        self.c_puct = 10  # 5
        self.buffer_size = 10000
        self.batch_size = 8  # mini-batch size for training  512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 10000#1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        #self_added
        self.buffer_no_extend = False
        #GB1 特有
        self.collected_seqs_index_set = set()
        # self.combo_to_index = combo_index_map
        # self.first_round_index = first_round_index
        self.update_predictor = 0
        self.updata_predictor_index_set = set()
        self.collected_seqs = set()###gluc
        self.seqs_and_fitness = []####gluc
        # GB1 特有
        self.last_tmp_set_len = 0
        #self_added
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.seq_len,
                                                   self.vocab_size,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.seq_len,
                                                   self.vocab_size)
        self.mcts_player = MCTSMutater(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        #
        self.update_predictor = 0
        #
        self.buffer_no_extend = False
        for i in range(n_games):
            play_data, seqs_and_fitness = self.mutate.start_mutating(self.mcts_player,
                                                          temp=self.temp)    #winner,
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            if self.episode_len == 0:
                self.buffer_no_extend = True
            # augment the data
            #play_data = self.get_equi_data(play_data)
            else:
                self.data_buffer.extend(play_data)
                s_f_list = list(seqs_and_fitness)
                #last_tmp_set_len = 0
                for seq, fitness ,frag_seq in s_f_list:
                    if seq not in ep_start_pool and seq not in self.collected_seqs:
                        self.collected_seqs.add(seq)
                        self.seqs_and_fitness.append({"sequence": seq, "fitness": fitness, "frag":frag_seq})

                    #self.last_tmp_set_len = len(tmp_set)
                # self.collect_seqs_set.union(gen_seqs)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy


    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))

                if len(self.seqs_and_fitness) >= 150:
                    saved_flag_4 = 1
                    #and saved_flag != 1: #100
                    df = pd.DataFrame(self.seqs_and_fitness)
                    df.to_csv(r"./output_design/MCTS_generated_gluc_1.csv", index=False)
                    print("saving seqs.............")
                    break

                # if len(self.updata_predictor_index_set) == 384:
                #     list_384 = []
                #     fit_384 = []
                #     index_to_combo = dict(zip(combo_to_index.values(), combo_to_index.keys()))
                #     for seq in list(self.updata_predictor_index_set):
                #         combo_tmp = index_to_combo[seq]
                #         list_384.append(combo_tmp)
                #         fit_384.append(combo_to_fitness[combo_tmp])
                #     df = pd.DataFrame({"AACombo":list_384, "Fitness": fit_384})
                #     #df.to_csv(r"E:\project_s\MCTS_Mutate_GB1\GB1\GP\MCTS_generated_384_GB1_{}.csv".format(self.round),index=False)
                #     df.to_csv(r"E:\project_s\MCTS_Mutate_GB1\georgiev\gp\MCTS_generated_384_PhoQ_{}.csv".format(self.round),
                #               index=False)
                #     #df.to_csv(r"E:\project_s\MCTS_Mutate_GB1\onehot\PhoQ\gp\MCTS_generated_384_PhoQ_{}.csv".format(self.round),index=False)
                #     #df.to_csv(r"E:\project_s\MCTS_Mutate_GB1\MCTS_generated_384_12.csv", index=False)
                #     current_time = int(time.time())
                #     localtime = time.localtime(current_time)
                #     dt = time.strftime('%Y:%m:%d %H:%M:%S', localtime)
                #     print(dt)  #
                #     break

                if len(self.data_buffer) > self.batch_size and self.buffer_no_extend == False:
                    loss, entropy = self.policy_update()
                
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Simple_cnn')
    
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU")
    parser.add_argument("--epochs", type=int, default=26,
                        help="number of training epochs")
    parser.add_argument("--seq_len", type=int, default=4,
                        help="protein len")
    parser.add_argument("--alphabet_len", type=int, default=0,
                        help="alphabet len")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="batch size")
    args = parser.parse_args()
    #combo_list = ["QMGE", "AMGH", "TGCN", "SMAL", "YGAG", "GDAS", "WEIQ", "EYFY", "EKVH"]
    #MODEL_PATH = r'E:\project_s\MCTS_Mutate_GB1\georgiev\cnn\checkpoint\CNN_PhoQ_checkpoint.tar'

    # cluster
    #df_seq = pd.read_csv(r"D:\数据\Gluc荧光素酶\gluc_model_input_data_2.csv")
    df_seq = pd.read_csv(r"./input_file/gluc_model_input_data_new.csv")

    sequences = df_seq["seqs"].values
    Fitness = df_seq["fitness"].values
    Fitness = Fitness / Fitness.max()
    # max start
    seq_to_fit = dict(zip(sequences, Fitness))
    first_round_d = sorted(seq_to_fit.items(), key=lambda x: x[1], reverse=True)  # [:40]
    ep_start_pool = [k for k, v in first_round_d]
    # max start
    # random start
    rand_seqs = list(sequences)
    random.shuffle(rand_seqs)
    # random start
    #start_pool = list(sequences)
    # start_pool = [k for k, v in first_round_d]
    #
    # seed = 100
    # first_round_index = []
    # for cluster_id in range(len(Index)):
    #     first_round_index.extend(SEQ_index[cluster_id])
    # first_round_index_set = set(first_round_index)
    # cluster
    #cnn_model = cnn_trainer(start_pool, Fitness, args)
    seq_list = list(sequences)
    Features = []
    for i in range(len(seq_list)):
        variant=seq_list[i]
        feature=feature_single(variant)
        Features.append(feature)
    # if len(Features.shape) == 3:
    #     features = np.reshape(Features, [Features.shape[0], Features.shape[1] * Features.shape[2]])
    model_gp = train_regr_predictor(Features, Fitness, seed)
    # cnn_trainer(AACombo, Fitness, first_round_index, args)
    # cnn_model = CNN(
    #     args.seq_len,
    #     args.alphabet_len,
    # )
    # cnn_model.load_state_dict(torch.load(MODEL_PATH))
    # starting_seq = SEQ_list[Fit_list.index(max(Fit_list))]
    training_pipeline = TrainPipeline(
        #ep_start_pool,  # max start
        rand_seqs, # rand start
        AAS,
        model_gp,
        # combo_to_feature,
        # combo_to_index,
        # first_round_index_set,
        # c_round,
        trust_radius=15,

    )
    training_pipeline.run()

    # for co in combo_list:
    #     c_one_hot = string_to_one_hot(co, AAS)
    #     one_hots = torch.from_numpy(c_one_hot)
    #     one_hots = one_hots.unsqueeze(0)
    #     one_hots = one_hots.to(torch.float32)
    #     with torch.no_grad():
    #         inputs = one_hots
    #         inputs = inputs.permute(0, 2, 1)
    #         # print('输入为：',inputs)
    #         outputs = cnn_model(inputs)
    #         outputs = outputs.squeeze()
    #     print(outputs)





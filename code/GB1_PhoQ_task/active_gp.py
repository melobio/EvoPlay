from __future__ import print_function
import argparse
import random
from collections import defaultdict, deque
from sequence_env_gp import Seq_env, Mutate
#from mcts_pure import MCTSPlayer as MCTS_Pure
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

def train_regr_predictor(features, fitness, Seq_index, seed):
    X_GP = features[Seq_index]
    Y_GP = fitness[Seq_index]
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

# def raw_to_features(c_list, f_list):
#
#     seq_np = np.array(
#         [string_to_one_hot(seq, AAS) for seq in c_list]
#     )
#
#     labels = torch.from_numpy(np.array(f_list))
#     labels = labels.to(torch.float32)
#
#     #normalize label
#     # max = labels.max()
#     # min = labels.min()
#     # factor = labels.add(-min)
#     # labels = factor.mul(1 / (max - min))
#     #normalize
#     one_hots = torch.from_numpy(seq_np)
#     one_hots = one_hots.to(torch.float32)
#     # print(one_hots)
#     return one_hots, labels
#
# class MyDataset(data.Dataset):
#     def __init__(self, sequences, labels):
#         self.sequences = sequences
#         self.labels = labels
#
#     def __getitem__(self, index):#返回的是tensor
#         seq, target = self.sequences[index], self.labels[index]
#         return seq, target
#
#     def __len__(self):
#         return len(self.sequences)

#CNN
# class CNN(nn.Module):
#     """predictor network module"""
#
#     def __init__(
#         self,
#         seq_len,
#         #alphabet,
#         alphabet_len,
#     ):
#         super(CNN, self).__init__()
#         self.board_width = seq_len
#         self.board_height = alphabet_len
#         # conv layers
#         self.conv1 = nn.Conv1d(20, 32, kernel_size=3, padding=1) #
#         # self.conv1 = nn.Conv1d(20, 32, kernel_size=3, padding=1)  #
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1) #, padding=0
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1) # , padding=0
#         self.conv4 = nn.Conv1d(128, 2, kernel_size=1)
#
#         # self.maxpool2 = nn.AdaptiveMaxPool1d(1)
#         # self.act_fc = nn.Linear(2 * seq_len * alphabet_len,
#         #                          seq_len * alphabet_len)
#         self.val_fc1 = nn.Linear(2* seq_len, 64)  # * alphabet_len
#   # * alphabet_len
#         self.dropout = nn.Dropout(p=0.25)
#         self.val_fc2 = nn.Linear(64, 1)
#
#     def forward(self, input):
#         x = F.relu(self.conv1(input))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x_act = F.relu(self.conv4(x))
#
#         #x_score_1 = x_act.view(-1, 2 * self.board_width * self.board_height)
#         x_score_1 = x_act.view(x_act.shape[0], -1)
#         #两层全连接层
#         x_score_2 = F.relu(self.val_fc1(x_score_1))
#         x_score_2 = self.dropout(x_score_2)
#         #x_score_3 = F.relu(self.val_fc2(x_score_2))
#         x_score_3 = F.tanh(self.val_fc2(x_score_2))
#
#         return x_score_3
#CNN

# def cnn_trainer(combo, fitness, Seq_index, args):
#     combo_list = combo[Seq_index]
#     fitness_list = fitness[Seq_index]
#
#     one_hots, labels = raw_to_features(combo_list, fitness_list)
#     seq_dataset = MyDataset(one_hots, labels)
#     train_loader = DataLoader(
#         seq_dataset, batch_size=args.batch_size, shuffle=True)
#     model = CNN(
#         args.seq_len,
#         args.alphabet_len,
#
#     )
#     optimizer = optim.Adam(model.parameters())
#     # optimizer = optim.Adam(model.parameters(),
#     #                        weight_decay=args.weight_decay)
#     for epoch in range(args.epochs):
#         for i, batch in enumerate(train_loader):
#             inputs = batch[0]
#             inputs = inputs.permute(0, 2, 1)  # conv1d 要求 输入顺序batch, embedding_dim，max_len
#             labels = batch[1]
#
#             optimizer.zero_grad()
#             # set_learning_rate(optimizer, args.learning_rate) #加了区别不大
#
#             logits = model(inputs)
#             # lr = param_group['lr']
#             logits = logits.squeeze()
#             loss = F.mse_loss(logits, labels)
#             loss.backward()
#             optimizer.step()
#         print("epoch:{}__loss:{}".format(epoch, loss))
#     PATH = r'E:\project_s\MCTS_Mutate_GB1\georgiev\cnn\checkpoint\CNN_PhoQ_checkpoint.tar'
#     torch.save(model.state_dict(), PATH)
#     return model

###
#train rgre model

AAS = "ILVAGMFYWEDQNHCRKSTP"

class TrainPipeline():
    def __init__(self, start_seq_pool, alphabet, model, combo_feature_map, combo_index_map, first_round_index, round, trust_radius, init_model=None): #init_model=None  feature_list,
        # params of the board and the game
        self.seq_len = len(start_seq_pool[0])
        self.vocab_size = len(alphabet)
        self.n_in_row = 4

        self.round = round
        #self.combo_index_map = combo_index_map
        # index_combo_map = {v: k for k, v in combo_index_map.items()}
        # first_round_combo = [index_combo_map[ind] for ind in first_round_index]
        first_round_combo = start_seq_pool
        self.seq_env = Seq_env(
            self.seq_len,
            alphabet,
            model,
            start_seq_pool,
            #combo_feature_map,
            #first_round_index,
            #combo_index_map,
            first_round_combo,
            #feature_list,
            trust_radius)  #n_in_row=self.n_in_row
        self.mutate = Mutate(self.seq_env)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 30  # num of simulations for each move 400
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
        self.combo_to_index = combo_index_map
        self.first_round_index = first_round_index
        self.update_predictor = 0
        self.updata_predictor_index_set = set()
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
            play_data, gen_seqs = self.mutate.start_mutating(self.mcts_player,
                                                          temp=self.temp)    #winner,
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            if self.episode_len == 0:
                self.buffer_no_extend = True
            # augment the data
            #play_data = self.get_equi_data(play_data)
            else:
                self.data_buffer.extend(play_data)
                #last_tmp_set_len = 0
                for seq in gen_seqs:
                    try:
                        self.collected_seqs_index_set.add(self.combo_to_index[seq])
                    except:
                        continue
                    tmp_set = self.collected_seqs_index_set.union(self.first_round_index)
                    print("tmp_set len: {}".format(len(tmp_set)))
                    if len(tmp_set) == 192 or len(tmp_set) == 288 and len(tmp_set)!= self.last_tmp_set_len:  #这个地方要优化，模型避免二次训练   or len(tmp_set) == 384
                        self.update_predictor =1
                        self.updata_predictor_index_set = tmp_set
                    if len(tmp_set) == 384:
                        self.updata_predictor_index_set = tmp_set

                    self.last_tmp_set_len = len(tmp_set)
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

                if self.update_predictor ==1:
                    print("predictor updating")
                    if self.updata_predictor_index_set:
                        print("collected seqs: {}".format(len(self.updata_predictor_index_set)))
                        seq_index_list = list(self.updata_predictor_index_set)
                        update_model = train_regr_predictor(features, Fitness, list(self.updata_predictor_index_set), seed)
                        #update_model = train_regr_predictor(AACombo, Fitness, list(self.updata_predictor_index_set), seed)

                        update_predictor_dict = dict(zip(AACombo[seq_index_list], Fitness[seq_index_list]))
                        update_round_d = sorted(update_predictor_dict.items(), key=lambda x: x[1],
                                               reverse=True)  # [:40]
                        #modified
                        self.seq_env.start_seq_pool = [k for k, v in update_round_d]
                        # new_start_seq_pool = [k for k, v in update_round_d]
                        # for removed_seq in self.mutate.remove_list:
                        #     if removed_seq in new_start_seq_pool:
                        #         new_start_seq_pool.remove(removed_seq)
                        #
                        # self.seq_env.start_seq_pool = new_start_seq_pool
                        # modified

                        self.seq_env.model = update_model
                        #self.update_predictor = 0

                if len(self.updata_predictor_index_set) == 384:
                    list_384 = []
                    fit_384 = []
                    index_to_combo = dict(zip(combo_to_index.values(), combo_to_index.keys()))
                    for seq in list(self.updata_predictor_index_set):
                        combo_tmp = index_to_combo[seq]
                        list_384.append(combo_tmp)
                        fit_384.append(combo_to_fitness[combo_tmp])
                    df = pd.DataFrame({"AACombo":list_384, "Fitness": fit_384})

                    # df.to_csv(r"./output_384_training_seqs/PhoQ/MCTS_generated_384_PhoQ_{}.csv".format(self.round),index=False)
                    df.to_csv(
                        r"./output_384_training_seqs/GB1/MCTS_generated_384_GB1_{}.csv".format(
                            self.round), index=False)

                    current_time = int(time.time())
                    localtime = time.localtime(current_time)
                    dt = time.strftime('%Y:%m:%d %H:%M:%S', localtime)
                    print(dt)  #
                    break

                if len(self.data_buffer) > self.batch_size and self.buffer_no_extend == False:
                    loss, entropy = self.policy_update()

        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Simple_cnn')
    # parser.add_argument("--data_dir", type=str, default="D:\\学习资料\\适配体项目\\代码\\GFP_train.txt",
    #                     help="train data dir")
    # parser.add_argument("--data_dir", type=str, default="C:\\MyFileAudit\\NGAL_40mers.txt",
    #                     help="train data dir")
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

    for c_round in range(1, 501):
        #cluster
        n_clusters = 30
        total_clusters = 30

        # PhoQ
        # features = np.load(r"./input_file/PhoQ_onehot_normalized.npy")
        # groundtruth = pd.read_excel(r"./input_file/PhoQ.xlsx")
        # PhoQ
        #GB1
        features = np.load(r"./input_file/GB1_onehot_normalized.npy")
        groundtruth = pd.read_excel(r"./input_file/GB1.xlsx")
        #GB1
        AACombo = groundtruth['Variants'].values
        Fitness = groundtruth['Fitness'].values
        Fitness = Fitness / Fitness.max()
        # print(features.shape)
        # print(len(Fitness))
        if len(features.shape) == 3:
            features = np.reshape(features, [features.shape[0], features.shape[1] * features.shape[2]])
        features_0 =  features
        features = features[0:len(Fitness)]
        # print(features.shape)
        Index = run_Clustering(features, n_clusters)
        Index = shuffle_index(Index)
        Prob = np.ones([n_clusters]) / n_clusters

        Fit_list = []
        SEQ_list = []
        Cluster_list = []
        #  store selected samples according to the cluster they belong to
        Fit = [[] for _ in range(len(Index))]
        SEQ = [[] for _ in range(len(Index))]
        SEQ_index = [[] for _ in range(len(Index))]
        num = 0
        num_first_round = 96
        while num < num_first_round:
            cluster_id = np.random.choice(np.arange(0, total_clusters), p=Prob)
            while len(Index[cluster_id]) == 0:
                Prob[cluster_id] = 0
                Prob = Prob / np.sum(Prob)
                cluster_id = np.random.choice(np.arange(0, total_clusters), p=Prob)
            Fit[cluster_id].append(Fitness[Index[cluster_id][0]])
            SEQ[cluster_id].append(AACombo[Index[cluster_id][0]])
            Fit_list.append(Fitness[Index[cluster_id][0]])
            SEQ_list.append(AACombo[Index[cluster_id][0]])
            SEQ_index[cluster_id].append(Index[cluster_id][0])
            Index[cluster_id] = np.delete(Index[cluster_id], [0])
            num += 1
        #
        combo_to_feature = dict(zip(AACombo, features))
        combo_to_index = dict(zip(AACombo, range(len(AACombo))))
        combo_to_fitness = dict(zip(AACombo, Fitness))
        combo_to_fitness_first_round = dict(zip(SEQ_list, Fit_list))
        first_round_d = sorted(combo_to_fitness_first_round.items(), key=lambda x: x[1], reverse=True) #[:40]
        start_pool = [k for k, v in first_round_d]
        #
        seed = 100
        first_round_index = []
        for cluster_id in range(len(Index)):
            first_round_index.extend(SEQ_index[cluster_id])
        first_round_index_set = set(first_round_index)
        #cluster
        model_gp = train_regr_predictor(features, Fitness, first_round_index, seed)
        starting_seq = SEQ_list[Fit_list.index(max(Fit_list))]
        training_pipeline = TrainPipeline(
            start_pool,  # starts["ed_10_wt"] ,gfp_wt_sequence, 0.1300  ,0.59, 0.005
            AAS,
            model_gp,
            combo_to_feature,
            combo_to_index,
            first_round_index_set,
            c_round,
            #feature_list,
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





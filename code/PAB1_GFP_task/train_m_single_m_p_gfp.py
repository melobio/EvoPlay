# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of EvoZero for GFP protein mutation
@author: Yi Wang
"""

from __future__ import print_function
import random
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sequence_env_m_p2 import Seq_env, Mutate
from mcts_alphaZero_mutate_expand_m_p_gfp import MCTSMutater
from p_v_net_torch import PolicyValueNet  # Pytorch
from p_v_net_3 import PolicyValueNet
from env_model import CNN
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Union
import sys
import datetime

data_dir = '/data/PAB1_GFP_data/GFP_237.txt'

pab1_wt_sequence = (
        "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    )
starts = {
        "start_seq": "SKGEELFTGVVPILVELDGDVNGHRFSVSGEGEGDATYGKPTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKARAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK" # noqa: E501
    }
AAS = "ILVAGMFYWEDQNHCRKSTP"

def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:

    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out
class MyDataset(data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, index):
        seq, target = self.sequences[index], self.labels[index]
        return seq, target

    def __len__(self):
        return len(self.sequences)
def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.
    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing
            a sequence.
        alphabet: Alphabet string (assigns each character an index).
    Returns:
        Sequence string representation of `one_hot`.
    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])

def raw_to_features(data_dir,part=1):
    fr = open(data_dir, "r")
    ll = fr.readlines()
    seq_list = []
    label_list = []
    print("part:{}".format(part))
    for i in range(1, len(ll)):
        tmp_list = ll[i].strip().split("\t")
        seq_list.append(tmp_list[0])
        label_list.append(float(tmp_list[1]))
        
    tem_part = int((len(seq_list)/10)*part)
    
    seq_list = seq_list[:tem_part]
    label_list =label_list[:tem_part]
    seq_np = np.array(
        [string_to_one_hot(seq, AAS) for seq in seq_list]
    )

    labels = torch.from_numpy(np.array(label_list))
    labels = labels.to(torch.float32)
    one_hots = torch.from_numpy(seq_np)
    one_hots = one_hots.to(torch.float32)
    return one_hots, labels

def train_cnn_predictor(data_dir,part=1):
    
    one_hots, labels = raw_to_features(data_dir,part)
    seq_dataset = MyDataset(one_hots, labels)
    index = [10,10,10,10,10,10,5,5,5,3,3]
    epochs = index[part]
    train_loader = DataLoader(
        seq_dataset, batch_size=128, shuffle=True)
    
    model = CNN(
        len(pab1_wt_sequence),
        len(AAS),).to('cuda')
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            inputs = batch[0]
            inputs = inputs.permute(0,2,1).to('cuda') 
            labels = batch[1].to('cuda')
            optimizer.zero_grad()
            
            logits = model(inputs)
            logits = logits.squeeze()
            loss = F.mse_loss(logits, labels)
            loss.backward()
            optimizer.step()
    print("train_loss:{}".format(loss))
    return model
class TrainPipeline():
    def __init__(self, start_seq, alphabet, model, trust_radius, init_model=None): #init_model=None
        self.seq_len = len(start_seq)
        self.vocab_size = len(alphabet)
        self.n_in_row = 4
        self.seq_env = Seq_env(
            self.seq_len,
            alphabet,
            model,
            start_seq,
            trust_radius)  #n_in_row=self.n_in_row
        self.mutate = Mutate(self.seq_env)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 200  # num of simulations for each move 400   1600
        self.c_puct = 10 #0.5  # 10
        self.buffer_size = 10000
        self.batch_size = 64  # mini-batch size for training  512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        #self_added
        self.buffer_no_extend = False
        #self_added
        #playout
        self.generated_seqs = []
        self.fit_list = []
        self.p_dict = {}
        self.m_p_dict = {}
        self.retrain_flag = False
        self.part = 2
        #playout
        #
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.seq_len,
                                                   self.vocab_size,
                                                   model_file=init_model,use_gpu=True)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.seq_len,
                                                   self.vocab_size,use_gpu=True)
        self.mcts_player = MCTSMutater(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        counts = len(self.generated_seqs)
        self.buffer_no_extend = False
        for i in range(n_games):
            play_data, seq_and_fit, p_dict = self.mutate.start_mutating(self.mcts_player,
                                                          temp=self.temp)    #winner,
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            self.p_dict = p_dict
            self.m_p_dict.update(self.p_dict)
            if self.episode_len == 0:
                self.buffer_no_extend = True
            else:
                self.data_buffer.extend(play_data)
                for seq, fit in seq_and_fit:  #alphafold_d
                    if seq not in self.generated_seqs:
                        self.generated_seqs.append(seq)
                        self.fit_list.append(fit)
                        if seq not in self.m_p_dict.keys():
                            self.m_p_dict[seq] = fit
                    
                        if len(self.generated_seqs)%10==0 and len(self.generated_seqs)>counts and self.part<=10:
                            self.retrain_flag=True
                       

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
    def policy_evaluate(self, n_games=10):

        current_mcts_player = MCTSMutater(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):   #
            winner = self.mutate.start_p_mutating(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        starttime = datetime.datetime.now() 
        #part =2
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if self.retrain_flag and self.part<=10:
                    print('train predictor again')

                    update_model = train_cnn_predictor(data_dir,self.part)
                    
                    self.seq_env.model = update_model
                    self.seq_env.model.eval()
                    self.part = self.part+1
                    self.retrain_flag = False
                if len(self.m_p_dict.keys()) >= 4000:
                    m_p_fitness = np.array(list(self.m_p_dict.values()))
                    m_p_seqs = np.array(list(self.m_p_dict.keys()))
                    df_m_p = pd.DataFrame(
                        {"sequence": m_p_seqs, "pred_fit": m_p_fitness})
                    df_m_p.to_csv(r"/code/PAB1_GFP_task/EvoPlay_gfp_generated_sequence_1.csv",index=False)
                    endtime = datetime.datetime.now() 
                    print('time cost：',(endtime-starttime).seconds)
                    sys.exit(0)
                if len(self.data_buffer) > self.batch_size and self.buffer_no_extend == False:
                    loss, entropy = self.policy_update()
        except KeyboardInterrupt:
            print('\n\rquit')

#model predict

if __name__ == '__main__':

    starttime = datetime.datetime.now() 
    model = train_cnn_predictor(data_dir)
    training_pipeline = TrainPipeline(
        starts["start_seq"], 
        AAS,
        model,
        trust_radius=100,
    )
    training_pipeline.run()

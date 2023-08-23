# -*- coding: utf-8 -*-
"""
@author: Yi Wang
"""

from __future__ import print_function
import random
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sequence_env_differ import Seq_env, Mutate
#from sequence_env import Seq_env, Mutate
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero_mutate import MCTSMutater
#from p_v_net_torch import PolicyValueNet  # Pytorch
from p_v_net_2 import PolicyValueNet
from env_model import CNN
import torch
from typing import List, Union


#pre_select genes
p_s_fr = open(r"E:\project_s\MCTS_Mutate_2_preSelect\GFP_wt_HighLikelihood_variant.txt")
ll = p_s_fr.readlines()
set1 = set()
for i in range(4,405):
    tm = ll[i].split("\t")[0][1:-1].split(", ")[0]
    set1.add(tm)
preSelect_list = list(set1)
preSelect_list = [int(tm) for tm in preSelect_list]
preSelect_list_1 = sorted(preSelect_list)
#pre_select genes

#要换成238
gfp_wt_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
starts = {
        "ed_10_wt": "MSKGEVLFTGVVPILVEMDGDVNGHKFSVSGEGEGDATYGKLTTKFTCTTGKLPVPWPTKVTTLSYRVQCFSRYPDVMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVQFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNIKRDCMVLLEFVTAAGITHGMDELYK",  # noqa: E501
        "ed_18_wt": "MSKGEHLFTGVVPILVELDGDVNGKKFSVSGEGQGDATYGKLTLKFICTTAKVHVPWCTLVTTLSYGVQCFSRYPDHMKQHDFFKGAMPEGYVQERTIFFKDIGNYKLRAEVKFEGDTLVNRIELKGIDFKEDGNIHGHKLEYNYNSQNVYIMASKQKNGIKVNFKIRLNIEDGSVQLAEHYQVNTPIGDFPVLLPDNHKLSAQSADSKDPNEKRDHMHLLEFVTAVGITHGMDELYK",  # noqa: E501
        "ed_31_wt": "MSKGEELFSGVQPILVELDGCVNGHKFSVSGEGEIDATYGKLTLKFICTTWKLPMPWPCLVTFGSYGVQCFSRYRDHPKQHDFFKSAVPEGYVQERTIFMKDDLLYKTRAEVKFEGLTLVNRIELKGKDFKEDGNILGHKLEYNYNSHCVYPMADWNKNWIKVNSKIRLPIEDGSVILADHYQQNTPIGDQPVLLPENHYLSTQSALSKDPEEKGDLMVLLEFVTAAGITHGMDELYK",  # noqa: E501
        "0.1300": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAIPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRVHMVLLEFVTAAGTTHGMDEQYK" ,
        "0.005": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAIPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMAVKQKDGIKVNFKNRHNIEDGSVRLADRYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK" ,
        "0.59": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRSEVKVEGDTLVNRIELKGIDLKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPSEKRDHMVLLEFVTAAGITHGMDELYK",
        "0.007": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTFFFKDDGNYKTRAEAKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    }
AAS = "ILVAGMFYWEDQNHCRKSTP"

def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:

    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out

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

def differ_dict(wt, mutate):
    d_dict = {}
    mutate_str_list = []
    if len(wt)!= len(mutate):
        return "error"
    for i in range(len(wt)):
        if wt[i] != mutate[i]:
            #print(str(i) + " " + str_1[i] + " " + str_2[i])
            d_dict[i] = mutate[i]
    if len(d_dict) > 0:
        for v in d_dict.values():
            mutate_str_list.append(v)
    mutate_str = "".join(mutate_str_list)

    return d_dict, mutate_str

def pre_select_dict(start_seq, pre_select_list):
    d_dict = {}
    mutate_str_list = []
    for i in range(len(pre_select_list)):
        d_dict[i] = pre_select_list[i]


    for ind in pre_select_list:
        mutate_str_list.append(start_seq[ind])
    # if len(d_dict) > 0:
    #     for v in d_dict.values():
    #         mutate_str_list.append(v)
    mutate_str = "".join(mutate_str_list)
    return d_dict, mutate_str

class TrainPipeline():
    def __init__(self, start_seq, alphabet, model, wt_seq, pre_select_list, trust_radius, preSelect=False, init_model=None): #init_model=None
        # params of the board and the game
        if preSelect == True:
            d_dict, mutate_str = pre_select_dict(start_seq, pre_select_list)
            self.starting_seq = mutate_str
            truncated = True
        else:
            self.starting_seq = start_seq
            truncated = False
        # if len(d_dict) >= 6:
        #     self.starting_seq = mutate_str
        #     truncated = True
        # else:
        #     self.starting_seq = start_seq
        #     truncated = False
        self.seq_len = len(self.starting_seq)
        self.vocab_size = len(alphabet)
        self.n_in_row = 4
        self.seq_env = Seq_env(
            self.seq_len,
            alphabet,
            model,
            self.starting_seq,
            trust_radius,
            start_seq,
            truncated,
            d_dict,
        )  #n_in_row=self.n_in_row
        self.mutate = Mutate(self.seq_env)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move 400, 800
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 8  # mini-batch size for training  512
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
        #GFP
        self.seqs_and_fitness = []
        #GFP
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
                for seq, fitness in s_f_list[1:]:
                 self.seqs_and_fitness.append({"sequence": seq, "fitness":fitness})

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
        for i in range(n_games):   #这个地方可能需要重新考虑一下
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
        saved_flag_1 = 0
        saved_flag_2 = 0
        saved_flag_3 = 0
        saved_flag_4 = 0
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                # saving result reqs
                if len(self.seqs_and_fitness) >= 50 and saved_flag_1 == 0:
                    saved_flag_1 = 1
                    # and saved_flag != 1: #100
                    df = pd.DataFrame(self.seqs_and_fitness)
                    df.to_csv(r"./result_seqs/MCTS_generated_gfp_selected_50.csv", index=False)
                    print("saving seqs.............")
                if len(self.seqs_and_fitness) >= 100 and saved_flag_2 == 0:
                    saved_flag_2 = 1
                    # and saved_flag != 1: #100
                    df = pd.DataFrame(self.seqs_and_fitness)
                    df.to_csv(r"./result_seqs/MCTS_generated_gfp_selected_100.csv", index=False)
                    print("saving seqs.............")
                if len(self.seqs_and_fitness) >= 150 and saved_flag_3 == 0:
                    saved_flag_3 = 1
                    # and saved_flag != 1: #100
                    df = pd.DataFrame(self.seqs_and_fitness)
                    df.to_csv(r"./result_seqs/MCTS_generated_gfp_selected_150.csv", index=False)
                    print("saving seqs.............")
                if len(self.seqs_and_fitness) >= 200 and saved_flag_4 == 0:
                    saved_flag_4 = 1
                    # and saved_flag != 1: #100
                    df = pd.DataFrame(self.seqs_and_fitness)
                    df.to_csv(r"./result_seqs/MCTS_generated_gfp_selected_200.csv", index=False)
                    print("saving seqs.............")
                # saving result reqs
                if len(self.data_buffer) > self.batch_size and self.buffer_no_extend == False:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                # if (i+1) % self.check_freq == 0:
                #     print("current self-play batch: {}".format(i+1))
                #     win_ratio = self.policy_evaluate()
                #     self.policy_value_net.save_model('./current_policy.model')
                #     if win_ratio > self.best_win_ratio:
                #         print("New best policy!!!!!!!!")
                #         self.best_win_ratio = win_ratio
                #         # update the best_policy
                #         self.policy_value_net.save_model('./best_policy.model')
                #         if (self.best_win_ratio == 1.0 and
                #                 self.pure_mcts_playout_num < 5000):
                #             self.pure_mcts_playout_num += 1000
                #             self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

#model predict

if __name__ == '__main__':
    MODEL_PATH = './gfp_checkpoint/CNN_GFP_checkpoint_19_0.0073_0.0519.tar'
    model2 = CNN(
        len(gfp_wt_sequence),
        len(AAS),

    )
    model2.load_state_dict(torch.load(MODEL_PATH))
    training_pipeline = TrainPipeline(
        starts["0.1300"], # starts["ed_10_wt"] ,gfp_wt_sequence, 0.1300  ,0.59, 0.005 ,0.007
        AAS,
        model2,
        gfp_wt_sequence,
        preSelect_list_1,
        trust_radius=15,
        preSelect=True,
    )
    training_pipeline.run()

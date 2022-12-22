# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import torch
import random
from typing import List, Union
import copy
import pandas as pd

# PhoQ
# features = np.load(r"./input_file/PhoQ_onehot_normalized.npy")
# groundtruth = pd.read_excel(r"./input_file/PhoQ.xlsx")

# GB1
features = np.load(r"./input_file/GB1_onehot_normalized.npy")
groundtruth = pd.read_excel(r"./input_file/GB1.xlsx")


AACombo = groundtruth['Variants'].values
Fitness = groundtruth['Fitness'].values
#Fitness = Fitness / Fitness.max()
if len(features.shape) == 3:
    features = np.reshape(features, [features.shape[0], features.shape[1] * features.shape[2]])
features = features[0:len(Fitness)]
combo_to_feature = dict(zip(AACombo, features))
#
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

def string_to_feature(string):
    seq_list = []
    seq_list.append(string)
    seq_np = np.array(
        [string_to_one_hot(seq, AAS) for seq in seq_list]
    )
    one_hots = torch.from_numpy(seq_np)
    one_hots = one_hots.to(torch.float32)
    return one_hots

class Seq_env(object):
    """sequence space for the env"""
    def __init__(self,
                 seq_len,
                 alphabet,
                 model,
                 starting_seq_pool,
                 #combo_feature_map,
                 #first_round_index,
                 #combo_index_map,
                 first_round_combo,
                 #feature_list,
                 trust_radus,
                 ):
        #自己加的变量
        #self.max_moves = trust_radus
        self.move_count = 0
        #自己加的变量
        self.seq_len = seq_len#self.width = int(kwargs.get('width', 8))
        self.vocab_size = len(alphabet)#self.height = int(kwargs.get('height', 8))
        #self.previous_fitness = -float("inf")
        self.alphabet = alphabet
        self.model = model
        #
        #self.start_seq_pool = starting_seq_pool #此处会改变  episode seqs
        self.start_seq_pool =starting_seq_pool
        #starting_seq = random.choice(self.start_seq_pool)
        starting_seq = self.start_seq_pool[0]
        self.starting_seq = starting_seq
        self.seq = starting_seq
        self.start_seq_pool.remove(starting_seq)
        #self.init_combo
        ####改
        self.init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32) #np.float32 是否需要改
        self.previous_init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.init_state_count = 0
        #self.previous_state = self.init_state
        #self.less_than_previous_count = []
        self.unuseful_move = 0
        self.states = {}
        #GB1特有
        #self.f_list = feature_list
        #self.combo_feature_map = combo_feature_map
        # GB1特有
        #self.episode_seqs = []
        #model 的代码
        #self.model.eval()   GB1 不需要
        #self.availables = list(range(self.seq_len * self.vocab_size))
        self.repeated_seq_ocurr = False
        #self.first_round_index = list(first_round_index)
        # self.combo_index_map = combo_index_map
        # self.index_combo_map = {v: k for k, v in self.combo_index_map.items()}
        # self.first_round_combo = [self.index_combo_map[ind] for ind in self.first_round_index]
        self.episode_seqs = first_round_combo

    def init_seq_state(self): #start_player=0
        #self.episode_seqs = []

        self.previous_fitness = -float("inf")
        self.move_count = 0
        #self.less_than_previous_count = []
        self.unuseful_move = 0
        #此处应该还会反复修改
        #
        self.repeated_seq_ocurr = False
        #
        #self._state = string_to_one_hot(self.starting_seq, self.alphabet).astype(np.float32)
        #start = string_to_one_hot(self.starting_seq, self.alphabet).astype(np.float32)
        self._state = copy.deepcopy(self.init_state)
            #加入变异的起始状态
        #initial plan 1 :random mutate
        # mut_len = 1 # mut_len = np.random.poisson(2) + 1
        # while True:
        #     for i in range(mut_len):
        #         pos = random.choice(list(range(start.shape[0])))
        #         pre_res = np.where(start[pos]==1)
        #         tmp_l = list(range(self.vocab_size))
        #         tmp_l.remove(pre_res[0])
        #         res = random.choice(tmp_l)
        #         self._state[pos] = 0
        #         self._state[pos, res] = 1
        #     #加入变异的起始状态
        #     #   起始序列的_state_fitness
        #
        #     #input = self.combo_feature_map[self.starting_seq]  # 固定starting seq时候用
        #     start_combo = one_hot_to_string(self._state, AAS)
        #     try:
        #         input = self.combo_feature_map[start_combo]
        #         break
        #     except:
        #         continue
        # # initial plan 1 :random mutate
        # # initial plan 2 :first round mutate
        # # input_combo = random.choice(self.first_round_combo)
        # # #input_combo = self.index_combo_map[input_index]
        # # input = self.combo_feature_map[input_combo]
        # # initial plan 2 :first round mutate
        #
        combo = one_hot_to_string(self._state, AAS)
        #看结果
        self.init_combo = combo
        # 看结果
        feature = []
        # try:
        #     for AA in combo:
        #         feature.append([self.f_list[0][AA], self.f_list[1][AA], self.f_list[2][AA], self.f_list[3][AA]])
        #     feature = np.asarray(feature)
        #     features = np.reshape(feature, [feature.shape[0] * feature.shape[1]])
        #     #input = self.combo_feature_map[combo]
        #     input = np.expand_dims(features, axis=0)
        #     outputs = self.model.predict(input)[0]
        # except:
        #     outputs = 0.0
        try:
            input = combo_to_feature[combo]
            input = np.expand_dims(input, axis=0)
            outputs = self.model.predict(input)[0]
        except:
            outputs = 0.0
        self._state_fitness = outputs
        #self.episode_seqs.append(one_hot_to_string(self._state, AAS))
        #   起始序列的_state_fitness
        # keep available moves in a list
        self.availables = list(range(self.seq_len * self.vocab_size))
        self.states = {}
        self.last_move = -1
        #
        self.previous_init_state = copy.deepcopy(self._state)
        #

    def current_state(self):

        square_state = np.zeros((self.seq_len, self.vocab_size))
        square_state = self._state
        return square_state.T
    def do_mutate(self, move):
        # #
        # self.repeated_seq_ocurr = False
        # #
        #self.previous_state = self._state
        ##
        self.previous_fitness = self._state_fitness
        self.move_count += 1
        self.availables.remove(move)
        pos = move // self.vocab_size
        res = move % self.vocab_size

        if self._state[pos, res] == 1:
            self.unuseful_move = 1
            self._state_fitness = 0.0
        else:
            self._state[pos] = 0
            self._state[pos, res] = 1

            # #episode seq check
            # if one_hot_to_string(self._state,AAS) in self.episode_seqs:
            #     self.unuseful_move = 1
            #     self._state_fitness = 0.0
            # else:
            #     self.episode_seqs.append(one_hot_to_string(self._state,AAS))
            #previous fitness
            combo = one_hot_to_string(self._state, AAS)
            #c_one_hot = string_to_one_hot(co, AAS)
            # feature = []
            # try:
            #     for AA in combo:
            #         feature.append([self.f_list[0][AA], self.f_list[1][AA], self.f_list[2][AA], self.f_list[3][AA]])
            #     feature = np.asarray(feature)
            #     features = np.reshape(feature, [feature.shape[0] * feature.shape[1]])
            #     # input = self.combo_feature_map[combo]
            #     input = np.expand_dims(features, axis=0)
            #     outputs = self.model.predict(input)[0]
            # except:
            #     outputs = 0.0
            try:
                input = combo_to_feature[combo]
                input = np.expand_dims(input, axis=0)
                outputs = self.model.predict(input)[0]
            except:
                outputs = 0.0

            self._state_fitness = outputs
        #
        current_seq = one_hot_to_string(self._state, AAS)
        # to be evaluated
        if current_seq in self.episode_seqs:
            self.repeated_seq_ocurr = True
            self._state_fitness = 0.0
            #self.availables.remove(move)
        else:
            self.episode_seqs.append(current_seq)
        # to be evaluated
        if self._state_fitness > self.previous_fitness:  # 0.6* 0.75*   #and not repeated_seq_ocurr
            # self._state_fitness = 0.0
            #self.availables.remove(move)
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0
        #

        self.last_move = move
        #



    def mutation_end(self):
        # to be evaluated
        if self.repeated_seq_ocurr == True:
            return True
        # to be evaluated
        # if self.move_count >= 20: # > 和 >= 的区别
        #     return True
        if self.unuseful_move == 1:
            return True
        if self._state_fitness < self.previous_fitness:  # 0.6* 0.75*
            #print("haha")
            return True

        return False

class Mutate(object):
    """mutating server"""

    def __init__(self, Seq_env, **kwargs):
        self.Seq_env = Seq_env
        self.remove_list = []

    def start_p_mutating(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.Seq_env.init_board(start_player)
        p1, p2 = self.Seq_env.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.Seq_env, player1.player, player2.player)
        while True:
            current_player = self.Seq_env.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.Seq_env)
            self.Seq_env.do_move(move)
            if is_shown:
                self.graphic(self.Seq_env, player1.player, player2.player)
            end, winner = self.Seq_env.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_mutating(self, mutater, is_shown=0, temp=1e-3):#mutater,
        """ start mutating using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        #
        if (self.Seq_env.previous_init_state == self.Seq_env.init_state).all():
            self.Seq_env.init_state_count += 1
        if self.Seq_env.init_state_count >= 10:  #10,6,7,5,8
            #new_start_seq = random.choice(self.Seq_env.start_seq_pool)
            new_start_seq = self.Seq_env.start_seq_pool[0]
            self.Seq_env.init_state = string_to_one_hot(new_start_seq, self.Seq_env.alphabet).astype(np.float32)
            self.Seq_env.start_seq_pool.remove(new_start_seq)
            self.Seq_env.init_state_count = 0
            self.remove_list.append(new_start_seq)
        #
        self.Seq_env.init_seq_state()
        print("起始序列：{}".format(self.Seq_env.init_combo))
        #p1, p2 = self.board.players
        states, mcts_probs, reward_z = [], [], [] #, current_players #, []
        generated_seqs = set()
        while True:
            move, move_probs = mutater.get_action(self.Seq_env,
                                                 temp=temp,
                                                 return_prob=1)
            if move:
                # store the data
                states.append(self.Seq_env.current_state())
                mcts_probs.append(move_probs)
                reward_z.append(self.Seq_env._state_fitness)
                # add generated seqs

                generated_seqs.add(one_hot_to_string(self.Seq_env._state, AAS))
                # perform a move
                self.Seq_env.do_mutate(move)
                print("move_fitness: %.16f\n" % (self.Seq_env._state_fitness))
                state_string = one_hot_to_string(self.Seq_env._state, AAS)
                print(state_string)
            end = self.Seq_env.mutation_end()
            if end:
                # 应不应该加 有待评估
                # states.append(self.Seq_env.current_state())  #la perdita ridurre piu senza questo
                # mcts_probs.append(move_probs)
                # reward_z.append(self.Seq_env._state_fitness)
                generated_seqs.add(one_hot_to_string(self.Seq_env._state, AAS))
                # 应不应该加 有待评估
                mutater.reset_Mutater()
                if is_shown:

                    print("Mutation end.")
                return zip(states, mcts_probs, reward_z), generated_seqs#winner,
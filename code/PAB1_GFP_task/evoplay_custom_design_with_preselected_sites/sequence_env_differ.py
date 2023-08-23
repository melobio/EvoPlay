# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import torch
import random
from typing import List, Union

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

def map_to_full_len_gfp(start_seq, d_dict, truncated_str):
    wt_list = list(start_seq)
    for k, v in d_dict.items():
        wt_list[v] = truncated_str[k]
    gfp_mutate_seq = "".join(wt_list)
    return gfp_mutate_seq

def string_to_feature(string):
    seq_list = []
    seq_list.append(string)
    seq_np = np.array(
        [string_to_one_hot(seq, AAS) for seq in seq_list]
    )
    one_hots = torch.from_numpy(seq_np)
    one_hots = one_hots.to(torch.float32)
    return one_hots

#def map_to_full_len_gfp():


class Seq_env(object):
    """sequence space for the env"""
    def __init__(self,
                 seq_len,
                 alphabet,
                 model,
                 starting_seq,
                 trust_radus,
                 wt_seq,
                 truncated,
                 map_dict,
                 ):
        #自己加的变量
        self.max_moves = trust_radus
        self.move_count = 0
        #自己加的变量
        self.seq_len = seq_len#self.width = int(kwargs.get('width', 8))
        self.vocab_size = len(alphabet)#self.height = int(kwargs.get('height', 8))

        self.alphabet = alphabet
        self.model = model
        self.starting_seq = starting_seq
        self.truncated = truncated
        self.wt = wt_seq
        self.map_dict = map_dict
        ####改
        self._state = string_to_one_hot(self.starting_seq, self.alphabet).astype(np.float32) #np.float32 是否需要改
        #self.less_than_previous_count = []
        self.unuseful_move = 0
        self.states = {}
        #self.episode_seqs = []
        #model 的代码
        self.model.eval()
        self.availables = list(range(self.seq_len * self.vocab_size))
        self.generated_seqs = []
        self.generated_seqs.append(self.starting_seq)
        self.from_truncated = wt_seq

    def init_seq_state(self): #start_player=0
        #self.episode_seqs = []
        self.previous_fitness = -float("inf")
        self.move_count = 0
        #self.less_than_previous_count = []
        self.unuseful_move = 0
        #此处应该还会反复修改
        #self._state = string_to_one_hot(self.starting_seq, self.alphabet).astype(np.float32)
        #加入变异的起始状态
        start = string_to_one_hot(self.starting_seq, self.alphabet).astype(np.float32)
        self._state = start
        # mut_len = np.random.poisson(2) + 1
        # for i in range(mut_len):
        #     pos = random.choice(list(range(start.shape[0])))
        #     pre_res = np.where(start[pos]==1)
        #     tmp_l = list(range(self.vocab_size))
        #     tmp_l.remove(pre_res[0])
        #     res = random.choice(tmp_l)
        #     self._state[pos] = 0
        #     self._state[pos, res] = 1
        # 加入变异的起始状态
        #   起始序列的_state_fitness
        #str_1 = one_hot_to_string(self._state, AAS)
        #self.episode_seqs.append(self.starting_seq)
        # if self.truncated == True:
        #     from_truncated = map_to_full_len_gfp(self.wt, self.map_dict)
        model_input = string_to_one_hot(self.wt, self.alphabet).astype(np.float32)
        # else:
        #     model_input = start
        # one_hots = torch.from_numpy(self._state)
        one_hots = torch.from_numpy(model_input)
        one_hots = one_hots.unsqueeze(0)
        one_hots = one_hots.to(torch.float32)
        # seq_dataset = MyDataset(one_hots, labels)
        with torch.no_grad():
            inputs = one_hots
            inputs = inputs.permute(0, 2, 1)
            # print('输入为：',inputs)
            outputs = self.model(inputs)
            outputs = outputs.squeeze()
        if outputs:
            self._state_fitness = outputs
        #   起始序列的_state_fitness
        # keep available moves in a list
        # self.availables = list(range(self.seq_len * self.vocab_size))
        self.states = {}
        self.last_move = -1

    def current_state(self):

        square_state = np.zeros((self.seq_len, self.vocab_size))
        square_state = self._state
        return square_state.T
    def do_mutate(self, move):
        #
        repeated_seq_ocurr = False
        #
        tmp_map_dict = {}
        self.previous_fitness = self._state_fitness
        self.move_count += 1
        #self.availables.remove(move)
        pos = move // self.vocab_size
        res = move % self.vocab_size

        if self._state[pos, res] == 1:
            self.unuseful_move = 1
            self._state_fitness = 0.0
        else:
            self._state[pos] = 0
            self._state[pos, res] = 1

            # if one_hot_to_string(self._state,AAS) in self.episode_seqs:
            #     self.unuseful_move = 1
            #     self._state_fitness = 0.0
            # else:
            #     self.episode_seqs.append(one_hot_to_string(self._state,AAS))
            #previous fitness
            if self.truncated == True:
                truncated_str = one_hot_to_string(self._state,AAS)
                # for ind, key in enumerate(self.map_dict):
                #     tmp_map_dict[key] = truncated_str[ind]
                self.from_truncated = map_to_full_len_gfp(self.wt, self.map_dict, truncated_str)
                model_input = string_to_one_hot(self.from_truncated, self.alphabet).astype(np.float32)
            else:
                model_input = self._state

            #one_hots_0 = self._state
            #one_hots = torch.from_numpy(one_hots_0)
            one_hots = torch.from_numpy(model_input)
            one_hots = one_hots.unsqueeze(0)
            one_hots = one_hots.to(torch.float32)
            with torch.no_grad():
                inputs = one_hots
                inputs = inputs.permute(0, 2, 1)
                # print('输入为：',inputs)
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
            if outputs:
                self._state_fitness = outputs
        #
        current_seq = one_hot_to_string(self._state, AAS)
        if current_seq in self.generated_seqs:
            repeated_seq_ocurr = True
            #self._state_fitness = 0.0
            self.availables.remove(move)
        else:
            self.generated_seqs.append(current_seq)
        if self._state_fitness < self.previous_fitness and not repeated_seq_ocurr:  # 0.6* 0.75*
            #self._state_fitness = 0.0
            self.availables.remove(move)
        self.last_move = move
        #



    def mutation_end(self):

        if self.move_count >= self.max_moves: # > 和 >= 的区别
            return True
        if self.unuseful_move == 1:
            return True
        if self._state_fitness < self.previous_fitness:  # 0.6* 0.75*
            return True

        return False

class Mutate(object):
    """mutating server"""

    def __init__(self, Seq_env, **kwargs):
        self.Seq_env = Seq_env

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
        self.Seq_env.init_seq_state()
        generated_seqs, seq_fitness = [], []
        #p1, p2 = self.board.players
        states, mcts_probs, reward_z = [], [], [] #, current_players #, []
        while True:
            move, move_probs = mutater.get_action(self.Seq_env,
                                                 temp=temp,
                                                 return_prob=1)
            if move:
                # store the data
                states.append(self.Seq_env.current_state())
                mcts_probs.append(move_probs)
                reward_z.append(self.Seq_env._state_fitness)
                #
                #
                # truncated_str = one_hot_to_string(self.Seq_env._state, AAS)
                # full_len_seq = map_to_full_len_gfp(self.wt, self.map_dict, truncated_str)
                generated_seqs.append(self.Seq_env.from_truncated)
                seq_fitness.append(float(self.Seq_env._state_fitness))
                #
                # perform a move
                self.Seq_env.do_mutate(move)
                print("move_fitness: %f\n" % (self.Seq_env._state_fitness))
                state_string = one_hot_to_string(self.Seq_env._state, AAS)
                print(state_string)
            end = self.Seq_env.mutation_end()
            if end:

                #
                generated_seqs.append(self.Seq_env.from_truncated)
                seq_fitness.append(float(self.Seq_env._state_fitness))
                #
                mutater.reset_Mutater()
                if is_shown:

                    print("Mutation end.")
                return zip(states, mcts_probs, reward_z), zip(generated_seqs, seq_fitness)  #winner,
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import torch
import random
from typing import List, Union
import copy

gfp_wt_sequence = (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVT"
        "TLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIE"
        "LKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNT"
        "PIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    )
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

def mutate_sequence(peptide_sequence, ex_list):
        '''Mutate the amino acid sequence randomly
        '''
        restypes = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V' ])
        seqlen = len(peptide_sequence)
        #searched_seqs = sequence_scores['sequence']
        #Mutate seq
        seeds = peptide_sequence
        #Go through a shuffled version of the positions and aas
        while True:
            seeds = peptide_sequence
            pi_s = np.random.choice(np.arange(seqlen), 3, replace=False)
            for pi in pi_s:
                aa = np.random.choice(restypes, replace=False)
                #new_s
                new_seq = seeds[:pi] + aa + seeds[pi + 1:]
                seeds = new_seq
            if new_seq not in ex_list:
                break
        return new_seq
    
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
                 starting_seq,
                 trust_radus,
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
        self.seq = starting_seq
        ####改
        self._state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32) #np.float32 是否需要改
        #self.less_than_previous_count = []
        self.init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.previous_init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.unuseful_move = 0
        self.states = {}
        self.episode_seqs = []
        self.episode_seqs.append(starting_seq)
        self.repeated_seq_ocurr = False
        self.init_state_count = 0
        #playout
        self.start_seq_exclude_list = []
        self.playout_dict = {}
        #playout
        #model 的代码
        self.model.eval()
    def init_seq_state(self): #start_player=0

        self.previous_fitness = -float("inf")
        self.move_count = 0
        #self.less_than_previous_count = []
        self.unuseful_move = 0
        #此处应该还会反复修改
        #self._state = string_to_one_hot(self.starting_seq, self.alphabet).astype(np.float32)
        #加入变异的起始状态
        self._state = copy.deepcopy(self.init_state)
        combo = one_hot_to_string(self._state, AAS)
        self.start_seq_exclude_list.append(combo)
        self.init_combo = combo
        #
        if combo not in self.episode_seqs:
            self.episode_seqs.append(combo)
        #
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
        #self.episode_seqs.append(one_hot_to_string(self._state,AAS))
        one_hots = torch.from_numpy(self._state)
        one_hots = one_hots.unsqueeze(0)
        one_hots = one_hots.to(torch.float32)
        # seq_dataset = MyDataset(one_hots, labels)
        with torch.no_grad():
            inputs = one_hots
            inputs = inputs.permute(0, 2, 1).to('cuda')
#             device = torch.device("cuda")
#             inputs = inputs.to(device)
            # print('输入为：',inputs)
            outputs = self.model(inputs)
#             outputs=outputs[0][0].cpu().numpy()[0]
            outputs = outputs.squeeze()
        if outputs:
            self._state_fitness = outputs
        #   起始序列的_state_fitness
        # keep available moves in a list
        self.availables = list(range(self.seq_len * self.vocab_size))
        #evo
        for i, a in enumerate(combo):
            self.availables.remove(self.vocab_size * i + AAS.index(a))
        for i, e_s in enumerate(self.episode_seqs):
            a_e_s = string_to_one_hot(e_s, AAS)
            a_e_s_ex = np.expand_dims(a_e_s, axis=0)
            if i == 0:
                nda = a_e_s_ex
            else:
                nda = np.concatenate((nda, a_e_s_ex), axis=0)

        c_i_s = string_to_one_hot(combo, AAS)
        for i, aa in enumerate(combo):
            tmp_c_i_s = np.delete(c_i_s, i, axis=0)
            for slice in nda:
                tmp_slice = np.delete(slice, i, axis=0)
                if (tmp_c_i_s == tmp_slice).all():
                    bias = np.where(slice[i] != 0)[0][0]
                    to_be_removed = self.vocab_size * i + bias
                    if to_be_removed in self.availables:
                        self.availables.remove(to_be_removed)
        #evo
        self.states = {}
        self.last_move = -1
        #
        self.previous_init_state = copy.deepcopy(self._state)
        #

    def current_state(self):

        square_state = np.zeros((self.seq_len, self.vocab_size))
        square_state = self._state
        return square_state.T
    def do_mutate(self, move, playout=0):
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

            combo = one_hot_to_string(self._state, AAS)
            if playout==0:
                if combo not in self.playout_dict.keys():
                    one_hots_0 = self._state
                    one_hots = torch.from_numpy(one_hots_0)
                    one_hots = one_hots.unsqueeze(0)
                    one_hots = one_hots.to(torch.float32)
                    with torch.no_grad():
                        inputs = one_hots
                        inputs = inputs.permute(0, 2, 1).to('cuda')
                        # print('输入为：',inputs)
#                         device = torch.device("cuda")
#                         inputs = inputs.to(device)
                        outputs = self.model(inputs)
                        outputs = outputs.squeeze()
                    if outputs:
                        self._state_fitness = outputs
                else:
                    self._state_fitness = self.playout_dict[combo]
                    #self.loss = 1/(1000*self._state_fitness)
            else:
                if combo not in self.playout_dict.keys():
                    one_hots_0 = self._state
                    one_hots = torch.from_numpy(one_hots_0)
                    one_hots = one_hots.unsqueeze(0)
                    one_hots = one_hots.to(torch.float32)
                    with torch.no_grad():
                        inputs = one_hots
                        inputs = inputs.permute(0, 2, 1).to('cuda')
#                         device = torch.device("cuda")
#                         inputs = inputs.to(device)
                        # print('输入为：',inputs)
                        outputs = self.model(inputs)
#                         outputs =outputs[0][0].cpu().numpy()[0]
                        outputs = outputs.squeeze()
                    if outputs:
                        self._state_fitness = outputs
                        self.playout_dict[combo] = outputs
                else:
                    #self._state_fitness = copy.deepcopy(self.playout_dict[combo])
                    self._state_fitness = self.playout_dict[combo]
                    #self.loss = 1/(1000*self._state_fitness)
        #
        current_seq = one_hot_to_string(self._state, AAS)
        if current_seq in self.episode_seqs:
            self.repeated_seq_ocurr = True
            self._state_fitness = 0.0
            # self.availables.remove(move)
        else:
            self.episode_seqs.append(current_seq)
        if self._state_fitness > self.previous_fitness:  # and not repeated_seq_ocurr:  # 0.6* 0.75*
            # self._state_fitness = 0.0
            # self.availables.remove(move)
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0
        #
        self.last_move = move
        #



    def mutation_end(self):
        if self.repeated_seq_ocurr == True:
            return True
        #
        if self.move_count >= self.max_moves: # > 和 >= 的区别
            return True
        #
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

        if (self.Seq_env.previous_init_state == self.Seq_env.init_state).all():
            self.Seq_env.init_state_count += 1
        if self.Seq_env.init_state_count >= 4:  #10,6,7,5,8
            print("随机起始更换****")
        #     #new_start_seq = random.choice(self.Seq_env.start_seq_pool)
        #     #new_start_seq = self.Seq_env.start_seq_pool[0]
            current_start_seq = one_hot_to_string(self.Seq_env.init_state, AAS)
            episode_seqs = copy.deepcopy(self.Seq_env.episode_seqs)
            playout_seqs = copy.deepcopy(list(self.Seq_env.playout_dict.keys()))
            e_p_list = list(set(episode_seqs + playout_seqs))
            #new_start_seq = mutate_sequence(current_start_seq, self.Seq_env.start_seq_exclude_list)
            new_start_seq = mutate_sequence(current_start_seq, e_p_list)
            self.Seq_env.init_state = string_to_one_hot(new_start_seq, self.Seq_env.alphabet).astype(np.float32)
            #self.Seq_env.start_seq_pool.remove(new_start_seq)  #这里会改变episode seq列表
            self.Seq_env.init_state_count = 0
 
        self.Seq_env.init_seq_state()
        print("起始序列：{}".format(self.Seq_env.init_combo))
        generated_seqs = []
        # alphafold_result = []
        fit_result = []
        play_seqs_list = []
        play_fit_list = []
        #p1, p2 = self.board.players
        states, mcts_probs, reward_z = [], [], [] #, current_players #, []
        while True:
            move, move_probs, play_seqs, play_losses = mutater.get_action(self.Seq_env,
                                                 temp=temp,
                                                 return_prob=1)
            self.Seq_env.playout_dict.update(mutater.m_p_dict)
            if move:
                # store the data
                states.append(self.Seq_env.current_state())
                mcts_probs.append(move_probs)
                reward_z.append(self.Seq_env._state_fitness)
                # perform a move
                self.Seq_env.do_mutate(move)
                generated_seqs.append(one_hot_to_string(self.Seq_env._state, AAS))
                # alphafold_result.append(self.Seq_env.alphfold_dict)
                fit_result.append(self.Seq_env._state_fitness)
                print("move_fitness: %f\n" % (self.Seq_env._state_fitness))
                print("episode_seq len: %d\n" % (len(self.Seq_env.episode_seqs)))
                print("Mmove & playout dict len: %d\n" % (len(self.Seq_env.playout_dict)))
                state_string = one_hot_to_string(self.Seq_env._state, AAS)
                print(state_string)
            end = self.Seq_env.mutation_end()
            if end:

                mutater.reset_Mutater()
                if is_shown:

                    print("Mutation end.")
                playout_dict = copy.deepcopy(self.Seq_env.playout_dict)
                return zip(states, mcts_probs, reward_z), zip(generated_seqs, fit_result), playout_dict  #winner,
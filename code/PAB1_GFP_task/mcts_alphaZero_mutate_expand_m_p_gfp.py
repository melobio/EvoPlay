# -*- coding: utf-8 -*-
from typing import List, Union, Dict, Optional
import numpy as np
import copy
import time
import torch
AAS = "ARNDCQEGHILKMFPSTWYV"
def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.

    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state, mp_dict):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        playout_seqs = []
        playout_fit = []
        state.playout_dict.update(mp_dict)
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_mutate(action, playout=1)
            playout_seq = one_hot_to_string(state._state, AAS)
            playout_seqs.append(playout_seq)
            playout_fit.append(state._state_fitness)

        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end = state.mutation_end()
        if not end:
            node.expand(action_probs)
        else:
            
            leaf_value = state._state_fitness

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)
        re_m_p_dict = copy.deepcopy(state.playout_dict)
        return playout_seqs, playout_fit, re_m_p_dict

    def get_move_probs(self, state, m_p_dict, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        play_seq_list = []
        play_fit_list = []
        g_m_p_dict = m_p_dict
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            #debug
            state_copy.playout = 1
            t_0 = time.time()
            play_seq, play_fit, mp_dict = self._playout(state_copy, g_m_p_dict)
            play_seq_list.extend(play_seq)
            play_fit_list.extend(play_fit)
            g_m_p_dict.update(mp_dict)


        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]

        if not act_visits:
            return [], []
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs, play_seq_list, play_fit_list, g_m_p_dict

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSMutater(object):
    """AI mutater based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.m_p_dict = {}

    def set_player_ind(self, p):
        self.player = p

    def reset_Mutater(self):
        self.mcts.update_with_move(-1)

    def get_action(self, Seq_env, temp=1e-3, return_prob=0):
        #sensible_moves = Seq_env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(Seq_env.seq_len*Seq_env.vocab_size)

        get_move_mp_dict = copy.deepcopy(self.m_p_dict)
        acts, probs, play_seqs, play_fitness, m_p_dict = self.mcts.get_move_probs(Seq_env, get_move_mp_dict, temp) #
        self.m_p_dict.update(m_p_dict)

        if acts:
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.03*np.ones(len(probs))) # 0.3 , 0.03
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
                print("AI move: %d\n" % (move))
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
                print("AI move: %d\n" % (move))

            if return_prob:
                return move, move_probs, play_seqs, play_fitness#, m_p_dict
            else:
                return move, play_seqs, play_fitness#, m_p_dict
        else:
            return [],[], play_seqs, play_fitness#, m_p_dict


    def __str__(self):
        return "MCTS {}".format(self.player)

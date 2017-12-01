#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'


from NFA_DFA_Module.DFA import DRA
from NFA_DFA_Module.DFA import LTL_plus
from MDP import MDP
from grid_graph import GridGraph

import numpy as np
from copy import deepcopy
from pprint import pprint

np.set_printoptions(linewidth=300)
np.set_printoptions(precision=3)

# Transition probabilities for each action in each cell (gross, explodes with
# the number of states).
prob_grid = {'North': np.array([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                                [0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
                                [0.0, 0.1, 0.9, 0.0, 0.0, 0.0],
                                [0.8, 0.0, 0.0, 0.1, 0.1, 0.0],
                                [0.0, 0.8, 0.0, 0.1, 0.0, 0.1],
                                [0.0, 0.0, 0.8, 0.0, 0.1, 0.1]]
                                ),
             'South': np.array([[0.1, 0.1, 0.0, 0.8, 0.0, 0.0],
                                [0.1, 0.0, 0.1, 0.0, 0.8, 0.0],
                                [0.0, 0.1, 0.1, 0.0, 0.0, 0.8],
                                [0.0, 0.0, 0.0, 0.9, 0.1, 0.0],
                                [0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
                                [0.0, 0.0, 0.0, 0.0, 0.1, 0.9]]
                                ),
             'East': np.array([[0.1, 0.8, 0.0, 0.1, 0.0, 0.0],
                               [0.0, 0.1, 0.8, 0.0, 0.1, 0.0],
                               [0.0, 0.0, 0.9, 0.0, 0.0, 0.1],
                               [0.1, 0.0, 0.0, 0.1, 0.8, 0.0],
                               [0.0, 0.1, 0.0, 0.0, 0.1, 0.8],
                               [0.0, 0.0, 0.1, 0.0, 0.0, 0.9]]
                               ),
             'West': np.array([[0.9, 0.0, 0.0, 0.1, 0.0, 0.0],
                               [0.8, 0.1, 0.0, 0.0, 0.1, 0.0],
                               [0.0, 0.8, 0.1, 0.0, 0.0, 0.1],
                               [0.1, 0.0, 0.0, 0.9, 0.0, 0.0],
                               [0.0, 0.1, 0.0, 0.8, 0.1, 0.0],
                               [0.0, 0.0, 0.1, 0.0, 0.8, 0.1]]
                               ),
             'Empty': np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
                                )
             }

shortest_paths = {frozenset([1, 2]): (1, 2),
                  frozenset([1, 3]): (1, 2, 3),
                  frozenset([1, 4]): (1, 4),
                  frozenset([1, 5]): (1, 2, 4),
                  frozenset([1, 6]): (1, 2, 3, 6),
                  frozenset([2, 3]): (2, 3),
                  frozenset([2, 4]): (2, 1, 4),
                  frozenset([2, 5]): (2, 5),
                  frozenset([2, 6]): (2, 3, 6),
                  frozenset([3, 4]): (3, 2, 1, 4),
                  frozenset([3, 5]): (3, 2, 5),
                  frozenset([3, 6]): (3, 6),
                  frozenset([4, 5]): (4, 5),
                  frozenset([4, 6]): (4, 5, 6),
                  frozenset([5, 6]): (5, 6)
                  }

# Entry point when called from Command line.
if __name__=='__main__':
    ##### Problem 2 - Configure MDP #####
    # For the simple 6-state gridworld, see slide 8 of Lecture 7, write the
    # specification automata for the following: visit all green cells and avoid
    # the red cell.
    # Shared atomic propositions:
    green3 = LTL_plus('green3')
    green4 = LTL_plus('green4')
    red = LTL_plus('red')
    empty = LTL_plus('E')
    # Where empty is the empty string/label/dfa-action.
    atom_prop = [green3, green4, red, empty]
    # Defined empty action for MDP incurrs a self loop.
    action_list = ['North', 'South', 'East', 'West', 'Empty']
    initial_state = '1'
    labels = {'1': empty,
              '2': empty,
              '3': green3,
              '4': green4,
              '5': red,
              '6': empty
              }
    # Note that input gamma is overwritten in DRA/MDP product method, so we'll
    # need to set it again later.
    grid_mdp = MDP(init=initial_state, action_list=action_list,
                   states=['1', '2', '3', '4', '5', '6'], prob=prob_grid,
                   gamma=0.9, AP=atom_prop, L=labels)

    ##### Configure EM inputs #####
    # Use a @ref GridGraph object to record, and seach for shortest paths
    # between two grid-cells.
    graph = GridGraph(shortest_paths)
    # Geodesic Gaussian Kernels, defined as Eq. 3.2 in Statistical Reinforcement
    # Learning, Sugiyama, 2015.
    ggk_sig = 1.0;
    kernel_centers = [1, 2, 3, 4, 5, 6]
    gg_kernel_func = lambda s_i, C_i: \
               np.exp( -graph.shortestPathLength(s_i, C_i)**2 / (2*ggk_sig**2) )
    # Note that we need to use a keyword style argument passing to ensure that
    # each lambda function gets its own value of C.
    K = [lambda s, C=cent: gg_kernel_func(s, C)
         for cent in kernel_centers]
    # It could be worth pre-computing all of the feature vectors for a small
    # grid...
    grid_mdp.addKernels(K)

    # Define the reward function for the grid_mdp. Get a reward when leaving
    # the winning statei 'q3' to 'q5'.
    pos_reward = {
                 'North': 0.0,
                 'South': 0.0,
                 'East': 0.0,
                 'West': 0.0,
                 'Empty': 1.0
                 }
    no_reward = {
                 'North': 0.0,
                 'South': 0.0,
                 'East': 0.0,
                 'West': 0.0,
                 'Empty': 0.0
                 }
    # 11/19/17. The goal is to reach state 3. All states are acceptable.  There
    # are no sinks.
    reward_dict = {}
    for state in grid_mdp.states:
        if '3' in state:
            # Winning state
            reward_dict[state] = pos_reward
        else:
            # No reward when leaving current state.
            reward_dict[state] = no_reward
    grid_mdp.reward = reward_dict
    # The following line should(?) force state 5 to be a sink state.
    # grid_mdp.setSinks('5')

    ##### SOLVE #####
    grid_mdp.solve(do_print=True, method='expectationMaximization')

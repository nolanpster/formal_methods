#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

from pprint import pprint

from TS import TS
from DFA_TS_product import DFATimesTS
from NFA_DFA_Module.DFA import DRA
from NFA_DFA_Module.DFA import LTL_plus
from MDP import MDP

import numpy as np
from copy import deepcopy


if __name__=='__main__':
    prob_dict = {'a': np.array([[0.7, 0.0, 0.3],
                                [0.3, 0.0, 0.7],
                                [0.0, 0.0, 1.0]]
                                ),
                 'b': np.array([[0.0, 0.6, 0.4],
                                [0.0, 0.5, 0.5],
                                [0.0, 0.0, 1.0]]
                                )
                 }
    reward = {'1': {
                        'a': 0.0,
                        'b': 0.2
                        },
              '2': {
                        'a': 0.2,
                        'b': 1.0
                        },
              '3': {
                        'a': 0.0,
                        'b': 0.0
                        }
              }

    quiz_mdp = MDP(init='1', actlist=['a','b'], states=['1', '2', '3'],
                   prob=prob_dict, gamma=1.0, reward=reward)
    quiz_mdp.gamma = 0.9
    quiz_mdp.solve(do_print=True) # Defaults to value iteration algorithm.
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
    actions = ['North', 'South', 'East', 'West', 'Empty']
    initial_state = '1'

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
                                   [0.0, 0.1, 0.0, 0.4, 0.1, 0.0],
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
    labels = {'1': empty,
              '2': empty,
              '3': green3,
              '4': green4,
              '5': red,
              '6': empty
              }
    grid_mdp = MDP(init=initial_state, actlist=actions,
                   states=['1', '2', '3', '4', '5', '6'], prob=prob_grid,
                   gamma=1, AP=atom_prop, L=labels)
    co_safe_dra = DRA(initial_state='q0', alphabet=[green3, green4, red, empty],
                      rabin_acc=[({'q3'},{})])
    # Empty transitions
    co_safe_dra.add_transition(empty, 'q0', 'q0')
    co_safe_dra.add_transition(empty, 'q1', 'q1')
    co_safe_dra.add_transition(empty, 'q2', 'q2')
    co_safe_dra.add_transition(empty, 'q3', 'q3')
    co_safe_dra.add_transition(empty, 'q4', 'q4')
    # Labeled transitions
    co_safe_dra.add_transition(green3, 'q0', 'q1')
    co_safe_dra.add_transition(green3, 'q1', 'q1')
    co_safe_dra.add_transition(green4, 'q1', 'q3')
    co_safe_dra.add_transition(green4, 'q0', 'q2')
    co_safe_dra.add_transition(green4, 'q2', 'q2')
    co_safe_dra.add_transition(green3, 'q2', 'q3')
    co_safe_dra.add_transition(empty, 'q3', 'q5')
    co_safe_dra.add_transition(empty, 'q5', 'q5')
    co_safe_dra.add_transition(red, 'q0', 'q4')
    co_safe_dra.add_transition(red, 'q1', 'q4')
    co_safe_dra.add_transition(red, 'q2', 'q4')
    # Once at sink, you're stuck there.
    #co_safe_dra.add_transition(red + 'Or' + ~red, 'q4')
    # Not adding a transition from 'q3' to 'q4' under red for simplicity. If we
    # get to 'q3' we win.
    co_safe_dra.toDot('visitGreensAndNoRed.dot')
    pprint(vars(co_safe_dra))
    game_mdp = MDP.productMDP(grid_mdp, co_safe_dra)
    # Define the reward function for the game_mdp. Get a reward when leaving
    # the winning state.
    pos_reward = {
                 'North': 1.0,
                 'South': 1.0,
                 'East': 1.0,
                 'West': 1.0,
                 }
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
    reward_dict = {}

    for state in game_mdp.states:
        if state in game_mdp.acc[0][0] and '5' not in state:
            # Winning state
            reward_dict[state] = pos_reward
        else:
            # No reward when leaving current state.
            reward_dict[state] = no_reward
    game_mdp.reward = reward_dict
    import pdb; pdb.set_trace()
    # Set all transition probabilities from a state containing 'q4' to take a
    # self loop with probability 1.
    game_mdp.findSinks('q4')
    game_mdp.findSinks('5')
    game_mdp.findSinks('q5')
    discounted_game_mdp = deepcopy(game_mdp)
    # Force this to be true since MDP constructor has a default gamma=0.9.
    discounted_game_mdp.gamma = 0.9
    discounted_game_mdp.solve(do_print=True)
    game_mdp.gamma = 1.0
    game_mdp.solve(do_print=True)

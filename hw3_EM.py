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

    ##### Add DRA for co-safe spec #####
    # Define a Deterministic (finitie) Raban Automata to match the sketch on
    # slide 7 of lecture 8. Note that state 'q4' is is the red, 'sink' state.
    co_safe_dra = DRA(initial_state='q0', alphabet=[green3, green4, red, empty],
                      rabin_acc=[({'q3'},{})])
    # Self-loops = Empty transitions
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
    co_safe_dra.add_transition(red, 'q0', 'q4')
    co_safe_dra.add_transition(red, 'q1', 'q4')
    co_safe_dra.add_transition(red, 'q2', 'q4')
    # If the DRA reaches state 'q3' we win. Therefore I do not define a
    # transition from 'q3' to 'q4'. Note that 'q4' is a sink state due to the
    # self loop.
    #
    # Also, I define a winning 'sink' state, 'q5'. I do this so that there is
    # only one out-going transition from 'q3' and it's taken only under the
    # empty action. This action, is the winning action. This is a little bit of
    # a hack, but it was the way that I thought of to prevent the system from
    # repeatedly taking actions that earned a reward.
    co_safe_dra.add_transition(empty, 'q3', 'q5')
    co_safe_dra.add_transition(empty, 'q5', 'q5')
    # Not adding a transition from 'q3' to 'q4' under red for simplicity. If we
    # get to 'q3' we win.
    if False:
        co_safe_dra.toDot('visitGreensAndNoRed.dot')
        pprint(vars(co_safe_dra))
    VI_game_mdp = MDP.productMDP(grid_mdp, co_safe_dra)
    # Define the reward function for the VI_game_mdp. Get a reward when leaving
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
    # Go through each state and if it is a winning state, assign it's reward
    # to be the positive reward dictionary. I have to remove the state
    # ('5', 'q3') because there are conflicting actions due to the label of '5'
    # being 'red'.
    reward_dict = {}
    for state in VI_game_mdp.states:
        if state in VI_game_mdp.acc[0][0] and '5' not in state:
            # Winning state
            reward_dict[state] = pos_reward
        else:
            # No reward when leaving current state.
            reward_dict[state] = no_reward
    VI_game_mdp.reward = reward_dict
    # Then I set up all sink states so all transition probabilities from a sink
    # states take a self loop with probability 1.
    VI_game_mdp.setSinks('q4')
    # If I uncomment the following line, all states at grid cell '5' no longer
    # build up any reward.
    VI_game_mdp.setSinks('5')
    VI_game_mdp.setSinks('q5')
    # @TODO Prune the MDP. Eg, state ('1', 'q3') is not reachable.
    EM_game_mdp = deepcopy(VI_game_mdp)
    # Set uniform initial distribution (this includes a uniform chance of being
    # at all states in the DRA which is wrong) -- need to fix!
    EM_game_mdp.setInitialProbDist()

    ##### SOLVE #####
    EM_game_mdp.solve(do_print=True, method='expectationMaximization')
    VI_game_mdp.solve(do_print=True, method='valueIteration')

    compare_to_decimals = 3
    VI_policy = VI_game_mdp.policy.copy()
    EM_policy = EM_game_mdp.policy.copy()
    policy_difference = EM_game_mdp.policy.copy()
    for state, action_dict in VI_policy.items():
        for act in action_dict.keys():
            VI_prob = round(VI_policy[state][act], compare_to_decimals)
            VI_policy[state][act] = VI_prob
            EM_prob = round(EM_policy[state][act], compare_to_decimals)
            EM_policy[state][act] = EM_prob
            policy_difference[state][act] = round(abs(VI_prob - EM_prob),
                                                  compare_to_decimals)
    print("Policy Difference:")
    pprint(policy_difference)

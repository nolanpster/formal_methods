#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'


from NFA_DFA_Module.DFA import DRA
from NFA_DFA_Module.DFA import LTL_plus
from MDP_EM.MDP_EM.MDP import MDP
from MDP_EM.MDP_EM.grid_graph import GridGraph

import os
import datetime
import pickle
import dill # For pickling lambda functions.
import numpy as np
from copy import deepcopy
from pprint import pprint

np.set_printoptions(linewidth=300)
np.set_printoptions(precision=3)


# Transition probabilities for each action in each cell (gross, explodes with
# the number of states).
#
# Row transition probabilites are:
# 0) Transition prob from Normal Grid cell to another normal grid cell.
# 1) Transition prob when adjacent to North Wall
# 2) Transition prob when adjacent to South Wall
# 3) Transition prob when adjacent to East Wall
# 4) Transition prob when adjacent to West Wall
# 5) Transition prob when in NE corner cell.
# 6) Transition prob when in NW corner cell.
# 7) Transition prob when in SE corner cell.
# 8) Transition prob when in SW corner cell.
#
# Column values are probabilities of ['Empty', 'north', 'south', 'east', 'west'] actions.
act_prob = {'North': np.array([[0.0, 0.8, 0.0, 0.1, 0.1],
                               [0.8, 0.0, 0.0, 0.1, 0.1],
                               [0.0, 0.8, 0.0, 0.1, 0.1],
                               [0.1, 0.8, 0.0, 0.0, 0.1],
                               [0.1, 0.8, 0.0, 0.1, 0.0],
                               [0.9, 0.0, 0.0, 0.0, 0.1],
                               [0.9, 0.0, 0.0, 0.1, 0.0],
                               [0.1, 0.8, 0.0, 0.0, 0.1],
                               [0.1, 0.8, 0.0, 0.1, 0.0]]
                               ),
            'South': np.array([[0.0, 0.0, 0.8, 0.1, 0.1],
                               [0.0, 0.0, 0.8, 0.1, 0.1],
                               [0.8, 0.0, 0.0, 0.1, 0.1],
                               [0.1, 0.0, 0.8, 0.0, 0.1],
                               [0.1, 0.0, 0.8, 0.1, 0.0],
                               [0.1, 0.0, 0.8, 0.0, 0.1],
                               [0.1, 0.0, 0.8, 0.1, 0.0],
                               [0.9, 0.0, 0.0, 0.0, 0.1],
                               [0.9, 0.0, 0.0, 0.1, 0.0]]
                               ),
            'East': np.array([[0.0, 0.1, 0.1, 0.8, 0.0],
                              [0.1, 0.0, 0.1, 0.8, 0.0],
                              [0.1, 0.1, 0.0, 0.8, 0.0],
                              [0.0, 0.1, 0.1, 0.8, 0.0],
                              [0.8, 0.1, 0.1, 0.0, 0.0],
                              [0.9, 0.0, 0.1, 0.0, 0.0],
                              [0.1, 0.0, 0.1, 0.8, 0.0],
                              [0.9, 0.1, 0.0, 0.0, 0.0],
                              [0.1, 0.1, 0.0, 0.8, 0.0]]
                              ),
            'West': np.array([[0.0, 0.1, 0.1, 0.0, 0.8],
                              [0.1, 0.0, 0.1, 0.0, 0.8],
                              [0.1, 0.1, 0.0, 0.0, 0.8],
                              [0.0, 0.1, 0.1, 0.0, 0.8],
                              [0.8, 0.1, 0.1, 0.0, 0.0],
                              [0.1, 0.0, 0.1, 0.0, 0.8],
                              [0.9, 0.0, 0.1, 0.0, 0.0],
                              [0.1, 0.1, 0.0, 0.0, 0.8],
                              [0.9, 0.1, 0.0, 0.0, 0.0]]
                              ),
            'Empty': np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0]]
                               )
            }

infer_act_prob = {'North': np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0]]
                                    ),
                 'South': np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0]]
                                    ),
                 'East': np.array([[0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0]]
                                   ),
                 'West': np.array([[0.0, 0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0]]
                                   ),
                 'Empty': np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0]]
                                    )
                 }

grid_map = np.array(range(0,6), dtype=np.int8).reshape([2,3])

shortest_paths = {frozenset([0, 1]): (0, 1),
                  frozenset([0, 2]): (0, 1, 2),
                  frozenset([0, 3]): (0, 3),
                  frozenset([0, 4]): (0, 1, 3),
                  frozenset([0, 5]): (0, 1, 2, 5),
                  frozenset([1, 2]): (1, 2),
                  frozenset([1, 3]): (1, 0, 3),
                  frozenset([1, 4]): (1, 4),
                  frozenset([1, 5]): (1, 2, 5),
                  frozenset([2, 3]): (2, 1, 0, 3),
                  frozenset([2, 4]): (2, 1, 4),
                  frozenset([2, 5]): (2, 5),
                  frozenset([3, 4]): (3, 4),
                  frozenset([3, 5]): (3, 4, 5),
                  frozenset([4, 5]): (4, 5)
                  }

# (Initial_state, next_state): Action
state_transition_actions = {
    (0,0): 'Empty', (0,1): 'East',                  (0,3): 'South',
    (1,0): 'West',  (1,1): 'Empty', (1,2): 'East',                  (1,4): 'South',
                    (2,1): 'West',  (2,2): 'Empty',                                 (2,5): 'South',
    (3,0): 'North',                                 (3,3): 'Empty', (3,4): 'East',
                    (4,1): 'North',                 (4,3): 'West',  (4,4): 'Empty', (4,5): 'East',
                                    (5,2): 'North',                 (5,4): 'West',  (5,5): 'Empty'
    }

mdp_obj_path = os.path.abspath('pickled_mdps')
data_path = os.path.abspath('pickled_episodes')
infered_mdps_path = os.path.abspath('pickled_inference')

def getOutFile(name_prefix='EM_MDP', dir_path=mdp_obj_path):
    # Dev machine returns UTC.
    current_datetime = datetime.datetime.now()
    formatted_time = current_datetime.strftime('_UTC%y%m%d_%H%M')
    # Filepath for mdp objects.
    full_file_path = os.path.join(dir_path, name_prefix + formatted_time)
    if not os.path.exists(os.path.dirname(full_file_path)):
        try:
            os.makedirs(os.path.dirname(full_file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return full_file_path

# Shared MDP Initialization Parameters.
green2 = LTL_plus('green2')
green3 = LTL_plus('green3')
red = LTL_plus('red')
empty = LTL_plus('E')
# Where empty is the empty string/label/dfa-action.
atom_prop = [green2, green3, red, empty]
# Defined empty action for MDP incurrs a self loop.
action_list = ['Empty', 'North', 'South', 'East', 'West']
initial_state = '0'
labels = {'0': empty,
          '1': empty,
          '2': green2,
          '3': green3,
          '4': red,
          '5': empty
          }

def makeGridMDPxDRA():
    ##### Problem 2 - Configure MDP #####
    # For the simple 6-state gridworld, see slide 8 of Lecture 7, write the
    # specification automata for the following: visit all green cells and avoid
    # the red cell.
    # Shared atomic propositions:
    # Note that input gamma is overwritten in DRA/MDP product method, so we'll
    # need to set it again later.
    grid_mdp = MDP(init=initial_state, action_list=action_list,
                   states=['0', '1', '2', '3', '4', '5'], act_prob=deepcopy(act_prob),
                   gamma=0.9, AP=atom_prop, L=labels, grid_map=grid_map)
    grid_mdp.init_set = grid_mdp.states

    ##### Add DRA for co-safe spec #####
    # Define a Deterministic (finitie) Raban Automata to match the sketch on
    # slide 7 of lecture 8. Note that state 'q4' is is the red, 'sink' state.
    co_safe_dra = DRA(initial_state='q0', alphabet=[green2, green3, red, empty],
                      rabin_acc=[({'q3'},{})])
    # Self-loops = Empty transitions
    co_safe_dra.add_transition(empty, 'q0', 'q0')
    co_safe_dra.add_transition(empty, 'q1', 'q1')
    co_safe_dra.add_transition(empty, 'q2', 'q2')
    co_safe_dra.add_transition(empty, 'q3', 'q3')
    co_safe_dra.add_transition(empty, 'q4', 'q4')
    # Labeled transitions
    co_safe_dra.add_transition(green2, 'q0', 'q1')
    co_safe_dra.add_transition(green2, 'q1', 'q1')
    co_safe_dra.add_transition(green3, 'q1', 'q3')
    co_safe_dra.add_transition(green3, 'q0', 'q2')
    co_safe_dra.add_transition(green3, 'q2', 'q2')
    co_safe_dra.add_transition(green2, 'q2', 'q3')
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
    VI_game_mdp.grid_map = grid_map
    # Define the reward function for the VI_game_mdp. Get a reward when leaving
    # the winning state 'q3' to 'q5'.
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
    # ('5', 'q3') because there are conflicting actions due to the label of '4'
    # being 'red'.
    reward_dict = {}
    for state in VI_game_mdp.states:
        if state in VI_game_mdp.acc[0][0] and '4' not in state:
            # Winning state
            reward_dict[state] = pos_reward
        else:
            # No reward when leaving current state.
            reward_dict[state] = no_reward
    VI_game_mdp.reward = reward_dict
    # Then I set up all sink states so all transition probabilities from a sink
    # states take a self loop with probability 1.
    VI_game_mdp.sink_act = 'Empty'
    VI_game_mdp.setSinks('q4')
    # If I uncomment the following line, all states at grid cell '5' no longer
    # build up any reward.
    #VI_game_mdp.setSinks('5')
    VI_game_mdp.setSinks('q5')
    # @TODO Prune the MDP. Eg, state ('1', 'q3') is not reachable.
    EM_game_mdp = deepcopy(VI_game_mdp)
    EM_game_mdp.setInitialProbDist(EM_game_mdp.init_set)

    ##### SOLVE #####
    VI_game_mdp.solve(do_print=True, method='valueIteration', write_video=False)
    EM_game_mdp.solve(do_print=True, method='expectationMaximization', write_video=False)

    compare_to_decimals = 3
    VI_policy = deepcopy(VI_game_mdp.policy)
    EM_policy = deepcopy(EM_game_mdp.policy)
    policy_difference = deepcopy(EM_game_mdp.policy)
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
    # Solved mdp.
    return EM_game_mdp


# Entry point when called from Command line.
if __name__=='__main__':
    # Program control flags.
    make_new_mdp = False
    gather_new_data = False
    perform_new_inference = False

    if make_new_mdp:
        mdp = makeGridMDPxDRA()
        mdp_file = getOutFile()
        with open(mdp_file, 'w+') as _file:
            print "Pickling mdp to {}".format(mdp_file)
            pickle.dump(mdp, _file)
    else:
        # Manually choose file here:
        mdp_file = os.path.join(mdp_obj_path, 'EM_MDP_UTC171207_0301')
        print "Loading file {}.".format(mdp_file)
        with open(mdp_file) as _file:
            mdp = pickle.load(_file)

    if gather_new_data:
        # Use policy to simulate and record results.
        #
        # Current policy E{T|R} 6.7. Start by simulating 10 steps each episode.
        num_episodes = 100
        steps_per_episode = 10
        run_histories = np.zeros([num_episodes, steps_per_episode], dtype=np.int8)
        for episode in range(num_episodes):
            # Create time-history for this episode.
            run_histories[episode, 0] = mdp.resetState()
            for t_step in range(1, steps_per_episode):
                run_histories[episode, t_step] = mdp.step()
        # Save sampled trajectories.
        history_file = getOutFile(os.path.basename(mdp_file)
                                  + ('_HIST_{}eps{}steps'.format(num_episodes, steps_per_episode)), data_path)
        with open(history_file, 'w+') as _file:
            print "Pickling Episode histories to {}.".format(history_file)
            pickle.dump(run_histories, _file)
    else:
        # Manually choose data to load here:
        history_file = os.path.join(data_path, 'EM_MDP_UTC171209_0401_HIST_100eps10steps_UTC171209_0401')
        print "Loading history data file {}.".format(history_file)
        with open(history_file) as _file:
            run_histories = pickle.load(_file)
    # Make sure we don't do anything with the loaded mdp other than gather data with it (we'll load it later for
    # comparison).
    mdp = None

    if perform_new_inference:
        # Solve for approximated observed policy.
        # Use a new mdp to model created/loaded one and a @ref GridGraph object to record, and seach for shortest paths
        # between two grid-cells.
        infer_mdp = MDP(init=initial_state, action_list=action_list,
                       states=['0', '1', '2', '3', '4', '5'], act_prob=deepcopy(act_prob),
                       gamma=0.9, AP=atom_prop, L=labels, grid_map=grid_map)
        infer_mdp.init_set = infer_mdp.states
        graph = GridGraph(grid_map=grid_map, neighbor_dict=infer_mdp.neighbor_dict)
        infer_mdp.graph = graph
        # Geodesic Gaussian Kernels, defined as Eq. 3.2 in Statistical Reinforcement
        # Learning, Sugiyama, 2015.
        ggk_sig = 1;
        kernel_centers = [0, 1, 2, 3, 4, 5]
        kernel_centers = [0, 2, 3, 5]
        #kernel_centers = [3, 4]
        # Note that this needs to be the same instance of `GridGraph` assigned to the MDP!
        gg_kernel_func = lambda s_i, C_i: np.exp( -graph.shortestPathLength(s_i, C_i)**2 / (2*ggk_sig**2) )
        # Note that we need to use a keyword style argument passing to ensure that
        # each lambda function gets its own value of C.
        K = [lambda s, C=cent: gg_kernel_func(s, C)
             for cent in kernel_centers]
        # It could be worth pre-computing all of the feature vectors for a small
        # grid...
        infer_mdp.addKernels(K)
        infer_mdp.precomputePhiAtState()

        # Infer the policy from the recorded data.
        infer_mdp.inferPolicy(histories=run_histories, do_print=True, use_precomputed_phi=True)
        infered_mdp_file = getOutFile(os.path.basename(history_file) + '_Policy', infered_mdps_path)
        with open(infered_mdp_file, 'w+') as _file:
            print "Pickling Infered Policy to {}.".format(infered_mdp_file)
            pickle.dump(infer_mdp, _file)
    else:
        # Manually choose data to load here:
        infered_mdp_file = os.path.join(infered_mdps_path,
            'EM_MDP_UTC171207_0301_HIST_100eps10steps_UTC171207_0301_Policy_UTC171207_0304')
        print "Loading infered policy data file {}.".format(infered_mdp_file)
        with open(infered_mdp_file) as _file:
            infer_mdp = pickle.load(_file)        # Reconsturct Policy with Q(s,a) = <theta, phi(s,a)>
    import pdb; pdb.set_trace()

    # Create plots for comparison.



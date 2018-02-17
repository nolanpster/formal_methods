#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'


from NFA_DFA_Module.DFA import DRA
from NFA_DFA_Module.DFA import LTL_plus
from MDP_EM.MDP_EM.MDP import MDP
from MDP_EM.MDP_EM.inference_mdp import InferenceMDP
import MDP_EM.MDP_EM.plot_helper as PlotHelp

import os
import datetime
import time
import pickle
import dill # For pickling lambda functions.
import csv
import numpy as np
from numpy import ma
from copy import deepcopy
from pprint import pprint
from collections import OrderedDict
import warnings
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

########################################################################################################################
# Numpy Print Options
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3)

########################################################################################################################
# Data save/load configuration
mdp_obj_path = os.path.abspath('pickled_mdps')
data_path = os.path.abspath('pickled_episodes')
infered_mdps_path = os.path.abspath('pickled_inference')
infered_statistics_path = os.path.abspath('pickled_inference_set_stats')

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

########################################################################################################################
# Transition Probability matricies

# Transition probabilities for each action in each cell  explodes with the number of states so we build the transition
# probabilites based on relative position based on grid walls.
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
                              [0.8, 0.1, 0.1, 0.0, 0.0],
                              [0.0, 0.1, 0.1, 0.8, 0.0],
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

########################################################################################################################
# Grid, number of agents, obstacle, label, action, initial and goal state configuration

grid_dim = [4, 4] # [num-rows, num-cols]
grid_map = np.array(range(0,np.prod(grid_dim)), dtype=np.int8).reshape(grid_dim)
states = [state for state in range(grid_map.size)]

# Atomic Proposition and labels configuration. Note that 'empty' is the empty string/label/dfa-action. The empty action
# for MDP incurrs a self loop.
green = LTL_plus('green')
red = LTL_plus('red')
empty = LTL_plus('E')
atom_prop = [green, red, empty]
labels = {state: empty for state in states}
labels[6] = red
labels[7] = red
labels[8] = red
labels[0] = green

# Starting and final states
initial_state = 15
goal_state = 0 # Currently assumess only one goal.

# Action options.
action_list = ['Empty', 'North', 'South', 'East', 'West']

# Set `solve_with_uniform_distribution` to True to have the initial distribution for EM and the history/demonstration
# generation start with a uniform (MDP.S default) distribution across the values assigned to MDP.init_set. Set this to
# _False_ to have EM and the MDP always start from the `initial_state` below.
solve_with_uniform_distribution = True

########################################################################################################################
def makeGridMDPxDRA(do_print=False):
    """
    @brief Configure the product MDP and DRA.

    Constructs an MDP based on the global variables. Then constructs a "4" state DRA, but two of the states are sink
    states.
    """
    # For the simple 6-state gridworld, see slide 8 of Lecture 7, write the specification automata for the following:
    # visit all green cells and avoid the red cell.
    #
    # Note shared atomic propositions:
    # Note that input gamma is overwritten in DRA/MDP product method, so we'll need to set it again later.
    grid_mdp = MDP(init=initial_state, action_list=action_list, states=states, act_prob=deepcopy(act_prob), gamma=0.9,
                   AP=atom_prop, L=labels, grid_map=grid_map)
    if solve_with_uniform_distribution:
        # Leave MDP.init_set unset (=None) if you want to solve the system from a single initial_state.
        grid_mdp.init_set = grid_mdp.states

    ##### Add DRA for co-safe spec #####
    # Define a Deterministic (finitie) Raban Automata to match the sketch on slide 7 of lecture 8. Note that state 'q2'
    # is is the red, 'sink' state.
    co_safe_dra = DRA(initial_state='q0', alphabet=atom_prop, rabin_acc=[({'q1'},{})])
    # Self-loops = Empty transitions
    co_safe_dra.add_transition(empty, 'q0', 'q0') # Initial state
    co_safe_dra.add_transition(empty, 'q2', 'q2') # Losing sink.
    co_safe_dra.add_transition(empty, 'q3', 'q3') # Winning sink.
    # Labeled transitions:
    # If the DRA reaches state 'q1' we win. Therefore I do not define a transition from 'q1' to 'q2'. Note that 'q2' and
    # 'q3' are a sink states due to the self loop.
    #
    # Also, I define a winning 'sink' state, 'q3'. I do this so that there is only one out-going transition from 'q1'
    # and it's taken only under the empty action. This action, is the winning action. This is a little bit of a hack,
    # but it was the way that I thought of to prevent the system from repeatedly taking actions that earned a reward.
    co_safe_dra.add_transition(green, 'q0', 'q1')
    co_safe_dra.add_transition(green, 'q1', 'q1') # State where winning action is available.
    co_safe_dra.add_transition(red, 'q0', 'q2')
    co_safe_dra.add_transition(empty, 'q1', 'q3')

    # Optional saving of DRA visualization file (can convert '.dot' file to PDF from terminal).
    if False:
        co_safe_dra.toDot('visitGreensAndNoRed.dot')
        pprint(vars(co_safe_dra))

    #### Create the Product MPDxDRA ####
    # Note that this isn't actually a "game" as define in Automata literature.
    VI_game_mdp = MDP.productMDP(grid_mdp, co_safe_dra)
    VI_game_mdp.grid_map = grid_map

    # Define the reward function for the VI_game_mdp. Get a reward when leaving
    # the winning state 'q1' to 'q3'.
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
        if state in VI_game_mdp.acc[0][0] and not VI_game_mdp.L[state]==red:
            # Winning state
            reward_dict[state] = pos_reward
        else:
            # No reward when leaving current state.
            reward_dict[state] = no_reward
    VI_game_mdp.reward = reward_dict

    # Then I set up all sink states so all transition probabilities from a sink
    # states take a self loop with probability 1.
    VI_game_mdp.sink_act = 'Empty'
    VI_game_mdp.setSinks('q3')
    VI_game_mdp.setSinks('q2')

    # @TODO Prune unreachable states from MDP.

    # Create a dictionary of reachable states for printing.
    policy_keys_to_print = deepcopy([(state[0], VI_game_mdp.dra.get_transition(VI_game_mdp.L[state], state[1])) for
                                     state in VI_game_mdp.states if 'q0' in state])
    if solve_with_uniform_distribution:
        initial_set = policy_keys_to_print
    else:
        initial_set = initial_state

    ##### SOLVE #####
    # To enable a solution of the MDP with multiple methods, copy the MDP, set the initial state likelihood
    # distributinos and then solve the MDPs.
    EM_game_mdp = deepcopy(VI_game_mdp)
    EM_game_mdp.setInitialProbDist(initial_set)
    VI_game_mdp.setInitialProbDist(initial_set)
    VI_game_mdp.solve(do_print=do_print, method='valueIteration', write_video=False,
                      policy_keys_to_print=policy_keys_to_print)
    EM_game_mdp.solve(do_print=do_print, method='expectationMaximization', write_video=False,
                      policy_keys_to_print=policy_keys_to_print)

    # Compare the two solution methods.
    policy_difference = MDP.comparePolicies(VI_game_mdp.policy, EM_game_mdp.policy, policy_keys_to_print,
                                            compare_to_decimals=3, do_print=do_print, compute_kl_divergence=True,
                                            reference_policy_has_augmented_states=True,
                                            compare_policy_has_augmented_states=True)

    return EM_game_mdp, VI_game_mdp, policy_keys_to_print, policy_difference

#######################################################################################################################
# Entry point when called from Command line.
if __name__=='__main__':
    # MDP solution/load options. If @c make_new_mdp is false load the @c pickled_mdp_file.
    make_new_mdp = True
    pickled_mdp_file = 'EM_MDP_UTC180217_1526'
    write_mdp_policy_csv = False

    # Demonstration history set of  episodes (aka trajectories) create/load options. If @c gather_new_data is false,
    # load the @c pickled_episodes_file. If @c gather_new_data is true, use @c num_episodes and @c steps_per_episode to
    # determine how large the demonstration set should be.
    gather_new_data = False
    num_episodes = 250
    steps_per_episode = 15
    pickled_episodes_file = 'EM_MDP_UTC180217_1526_HIST_250eps15steps_UTC180217_1526'

    # Perform/load policy inference options. If @c perform_new_inference is false, load the
    # @pickled_inference_mdps_file. The inference statistics files contain an array of L1-norm errors from the
    # demonstration policy.
    perform_new_inference = True
    pickled_inference_mdps_file = 'EM_MDP_UTC180217_1526_HIST_250eps15steps_UTC180217_1526_Policy_UTC180217_1526'
    load_inference_statistics = (not perform_new_inference) & True
    pickled_inference_statistics_file = \
        'EM_MDP_UTC180205_1024_HIST_250eps15steps_UTC180205_1032_Inference_Stats_UTC180205_1046'
    inference_method='default' # Default chooses gradient ascent. Other options: 'historyMLE', 'iterativeBayes'.

    # Gradient Ascent kernel configurations
    kernel_sigmas = [1]*6
    kernel_count_start = 6
    kernel_count_end = 5
    kernel_count_increment_per_set = -1
    kernel_set_sample_count = 1
    batch_size_for_kernel_set = 1

    # Plotting lags
    plot_all_grids = True
    plot_initial_mdp_grids = False
    plot_inferred_mdp_grids = False
    plot_new_phi = True
    plot_new_kernel = False
    plot_loaded_phi = False
    plot_loaded_kernel = False
    plot_inference_statistics = True
    plot_flags = [plot_all_grids, plot_initial_mdp_grids, plot_inferred_mdp_grids, plot_new_phi, plot_loaded_phi,
                  plot_new_kernel, plot_loaded_kernel, plot_inference_statistics]
    if plot_new_kernel and plot_loaded_kernel:
        raise ValueError('Can not plot both new and loaded kernel in same call.')
    if plot_new_kernel:
        raise NotImplementedError('option: plot_new_kernel doesn\'t work yet. Sorry, but plot_new_phi works!')
    if plot_new_phi and plot_loaded_phi:
        raise ValueError('Can not plot both new and loaded phi in same call.')

    # Configure kernel set iterations.
    num_kernels_in_set = np.arange(kernel_count_start, kernel_count_end, kernel_count_increment_per_set)
    num_kernel_sets = len(num_kernels_in_set)
    kernel_centers = {set_idx: frozenset(np.random.choice(len(states), num_kernels_in_set[set_idx] , replace=False)) for
                      set_idx in range(num_kernel_sets)}
    kernel_set_L1_err = np.empty([num_kernel_sets, kernel_set_sample_count])
    kernel_set_infer_time = np.empty([len(num_kernels_in_set), kernel_set_sample_count])

    if make_new_mdp:
        EM_mdp, VI_mdp, policy_keys_to_print, policy_difference = makeGridMDPxDRA(do_print=True)
        mdp_file = getOutFile()
        with open(mdp_file, 'w+') as _file:
            print "Pickling EM_mdp to {}".format(mdp_file)
            pickle.dump([EM_mdp, VI_mdp, policy_keys_to_print,policy_difference], _file)
    else:
        # Manually choose file here:
        mdp_file = os.path.join(mdp_obj_path, pickled_mdp_file)
        print "Loading file {}.".format(mdp_file)
        with open(mdp_file) as _file:
            EM_mdp, VI_mdp, policy_keys_to_print, policy_difference = pickle.load(_file)
    if write_mdp_policy_csv:
        diff_csv_dict = {key: policy_difference[key] for key in policy_keys_to_print}
        with open(mdp_file+'_Policy_difference.csv', 'w+') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in diff_csv_dict.items():
                for subkey, sub_value in value.items():
                    writer.writerow([key,subkey,sub_value])
        #EM_csv_dict = {key: EM_mdp.policy[key] for key in policy_keys_to_print}
        #with open(mdp_file+'_EM_Policy.csv', 'w+') as csv_file:
        #    writer = csv.writer(csv_file)
        #    for key, value in EM_csv_dict.items():
        #        for subkey, sub_value in value.items():
        #            writer.writerow([key,subkey,sub_value])
        #VI_csv_dict = {key: VI_mdp.policy[key] for key in policy_keys_to_print}
        #with open(mdp_file+'_VI_policy.csv', 'w+') as csv_file:
        #    writer = csv.writer(csv_file)
        #    for key, value in VI_csv_dict.items():
        #        for subkey, sub_value in value.items():
        #            writer.writerow([key,subkey,sub_value])

    # Choose which policy to use for demonstration.
    mdp = VI_mdp
    reference_policy_vec = mdp.getPolicyAsVec(policy_keys_to_print)

    if gather_new_data:
        # Use policy to simulate and record results.
        #
        # Current policy E{T|R} 6.7. Start by simulating 10 steps each episode.
        num_episodes = 250
        steps_per_episode = 15
        if mdp.num_states < np.iinfo(np.uint8).max:
            hist_dtype = np.uint8
        elif mdp.num_states < np.iinfo(np.uint16).max:
            hist_dtype = np.uint16
        elif mdp.num_states < np.iinfo(np.uint32).max:
            hist_dtype = np.uint32
        elif mdp.num_states < np.iinfo(np.uint64).max:
            hist_dtype = np.uint64
        else:
            raise ValueError('This MDP has {} states, that\'s not currently supported, I\'m surprised your code made '
                             'it this far...'.format(np.iinfo(np.uint64).max))
        run_histories = np.zeros([num_episodes, steps_per_episode], dtype=hist_dtype)
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
        history_file = os.path.join(data_path, pickled_episodes_file)
        print "Loading history data file {}.".format(history_file)
        with open(history_file) as _file:
            run_histories = pickle.load(_file)
        num_episodes = run_histories.shape[0]
        steps_per_episode = run_histories.shape[1]

    # Determine which states are goals or obstacles.
    normal_states = {state: True if label==empty else False for state, label in labels.items()}
    unique, starting_counts = np.unique(run_histories[:,0], return_counts=True)
    num_trials_from_state = {state:0 for state in states}
    num_trials_from_state.update(dict(zip(unique, starting_counts)))
    num_rewards_from_state = {state:0 for state in states}
    for run_idx in range(num_episodes):
        starting_state = run_histories[run_idx][0]
        final_state = run_histories[run_idx][-1]
        if final_state==goal_state:
            num_rewards_from_state[starting_state] += 1
    print("In this demonstration 'history' there are  {} episodes, each with {} moves.".format(num_episodes,
          steps_per_episode))
    for state in range(len(states)):
        reward_likelihood = float(num_rewards_from_state[state]) / float(num_trials_from_state[state]) if \
            num_trials_from_state[state] > 0 else np.nan
        print("State {}: Num starts = {}, Num Rewards = {}, likelihood = {}.".format(state,
                                                                                     num_trials_from_state[state],
                                                                                     num_rewards_from_state[state],
                                                                                     reward_likelihood))

    if plot_new_phi or  plot_new_kernel or perform_new_inference:
        tic = time.clock()
        # Solve for approximated observed policy.
        # Use a new mdp to model created/loaded one and a @ref GridGraph object to record, and seach for shortest paths
        # between two grid-cells.
        infer_mdp = InferenceMDP(init=initial_state, action_list=action_list, states=states,
                                 act_prob=deepcopy(act_prob), grid_map=grid_map, L=labels,
                                 gg_kernel_centers=kernel_centers[0],kernel_sigmas=kernel_sigmas)
        print 'Built InferenceMDP with kernel set:'
        print(kernel_centers[0])
        if not perform_new_inference and (plot_new_phi or plot_new_kernel):
            # Deepcopy the infer_mdp to another variable because and old inference will be loaded into `infer_mdp`.
            new_infer_mdp = deepcopy(infer_mdp)
    if perform_new_inference:
        # Infer the policy from the recorded data.

        # Precompute observed actions for all episodes. Should do this in a "history" class.
        if len(action_list) < np.iinfo(np.uint8).max:
            observation_dtype = np.uint8
        elif len(action_lists) < np.iinfo(np.uint16).max:
            observation_dtype  = np.uint16
        elif len(action_lists) < np.iinfo(np.uint32).max:
            observation_dtype  = np.uint32
        elif len(action_lists) < np.iinfo(np.uint64).max:
            observation_dtype  = np.uint64
        else:
            raise ValueError('This MDP has {} actions, that\'s not currently supported, I\'m surprised your '
                             'code made it this far...'.format(np.iinfo(np.uint64).max))
        observed_action_indeces = np.empty([num_episodes, steps_per_episode], dtype=observation_dtype)
        for episode in xrange(num_episodes):
            for t_step in xrange(1, steps_per_episode):
                this_state = run_histories[episode, t_step-1]
                next_state = run_histories[episode, t_step]
                observed_action = infer_mdp.graph.getObservedAction(this_state, next_state)
                observed_action_indeces[episode, t_step] = action_list.index(observed_action)

        if inference_method is not 'default':
            infer_mdp.inferPolicy(method=inference_method, histories=run_histories, do_print=True,
                                  reference_policy_vec=reference_policy_vec)
        else:
            if batch_size_for_kernel_set > 1:
                print_inference_iterations = False
            else:
                print_inference_iterations = True

            # Peform the inference batch.
            for kernel_set_idx in xrange(num_kernel_sets):

                batch_L1_err = np.empty([kernel_set_sample_count, batch_size_for_kernel_set])
                batch_infer_time = np.empty([kernel_set_sample_count, batch_size_for_kernel_set])
                for trial in xrange(kernel_set_sample_count):
                    trial_kernel_set = frozenset(np.random.choice(len(states), num_kernels_in_set[kernel_set_idx],
                                                 replace=False))
                    trial_kernel_set |= set([0])
                    print('Inference set {} has {} kernels:{}'.format(
                          (kernel_set_idx * kernel_set_sample_count) + trial, num_kernels_in_set[kernel_set_idx],
                          trial_kernel_set))
                    infer_mdp.buildKernels(trial_kernel_set)

                    batch_L1_err[trial], batch_infer_time[trial] = \
                        infer_mdp.inferPolicy(histories=run_histories,
                                              do_print=print_inference_iterations,
                                              use_precomputed_phi=True,
                                              dtype=np.float32,
                                              monte_carlo_size=batch_size_for_kernel_set,
                                              reference_policy_vec=reference_policy_vec,
                                              precomputed_observed_action_indeces=observed_action_indeces)
                kernel_set_L1_err[kernel_set_idx] = batch_L1_err.mean(axis=1)
                kernel_set_infer_time[kernel_set_idx] = batch_infer_time.mean(axis=1)
        toc = time.clock() -tic
        print 'Total time to infer policy{}: {} sec, or {} min.'.format(' set' if num_kernel_sets > 1 else '', toc,
                                                                        toc/60.0)
        infered_mdp_file = getOutFile(os.path.basename(history_file) + '_Policy', infered_mdps_path)
        with open(infered_mdp_file, 'w+') as _file:
            print "Pickling Infered Policy to {}.".format(infered_mdp_file)
            pickle.dump(infer_mdp, _file)
        if num_kernel_sets > 1:
            # Save data for inference sets
            infered_stats_file = getOutFile(os.path.basename(history_file) + '_Inference_Stats',
                                            infered_statistics_path)
            with open(infered_stats_file, 'w+') as _file:
                print "Pickling Inference Statistics to {}.".format(infered_stats_file)
                pickle.dump([kernel_set_L1_err, kernel_set_infer_time], _file)

    else:
        # Manually choose data to load here:
        infered_mdp_file = os.path.join(infered_mdps_path, pickled_inference_mdps_file)
        print "Loading infered policy data file {}.".format(infered_mdp_file)
        with open(infered_mdp_file) as _file:
            infer_mdp = pickle.load(_file)        # Reconsturct Policy with Q(s,a) = <theta, phi(s,a)>
    if write_mdp_policy_csv:
        infer_csv_dict = {key: infer_mdp.policy[key] for key in infer_mdp.policy.keys()}
        with open(infered_mdp_file+'.csv', 'w+') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in infer_csv_dict.items():
                for subkey, sub_value in value.items():
                    writer.writerow([key,subkey,sub_value])

    if load_inference_statistics:
        # Manually choose data to load here:
        infered_stats_file = os.path.join(infered_statistics_path, pickled_inference_statistics_file)
        print "Loading inference statistics data file {}.".format(infered_stats_file)
        with open(infered_stats_file) as _file:
           kernel_set_L1_err, kernel_set_infer_time = pickle.load(_file)

    # Remember that variable @ref mdp is used for demonstration.
    if len(policy_keys_to_print) == infer_mdp.num_states:
        infered_policy_L1_norm_error = MDP.getPolicyL1Norm(reference_policy_vec, infer_mdp.getPolicyAsVec())
        print('L1-norm between reference and inferred policy: {}.'.format(infered_policy_L1_norm_error))
    else:
        warnings.warn('Demonstration MDP and inferred MDP do not have the same number of states. Perhaps one was '
                      'loaded from an old file? Not printing policy difference.')

    if any(plot_flags):
        # Create plots for comparison. Note that the the `maze` array has one more row and column than the `grid` for
        # plotting purposes.
        maze = np.zeros(np.array(grid_dim)+1)
        for state, label in labels.iteritems():
            if label==red:
                grid_row, grid_col = np.where(grid_map==state)
                maze[grid_row, grid_col] = 2
            if label==green:
                grid_row, grid_col = np.where(grid_map==state)
                maze[grid_row, grid_col] = 1
        if red in labels.values():
            # Maximum value in maze corresponds to red.
            cmap = mcolors.ListedColormap(['white','green','red'])
        else:
            # Maximum value in maze corresponds to green.
            cmap = mcolors.ListedColormap(['white','green'])

    plot_policies = []
    only_use_print_keys = []
    titles = []
    kernel_locations = []
    if plot_all_grids:
        plot_policies.append(VI_mdp.policy)
        plot_policies.append(EM_mdp.policy)
        plot_policies.append(infer_mdp.policy)
        titles = ['Value Iteration', 'Expecation Maximization', 'Learned']
        only_use_print_keys = [True, True, False]
        kernel_locations = [None, None, infer_mdp.kernel_centers]
    elif plot_initial_mdp_grids:
        plot_policies.append(VI_mdp.policy)
        plot_policies.append(EM_mdp.policy)
        titles = ['Value Iteration', 'Expecation Maximization']
        kernel_locations = [None, None]
        only_use_print_keys = [True, True]
    if plot_inferred_mdp_grids and not plot_all_grids:
        plot_policies.append(infer_mdp.policy)
        titles.append('Learned')
        only_use_print_keys.append(False)
        kernel_locations.append(infer_mdp.kernel_centers)

    if plot_all_grids or plot_initial_mdp_grids or plot_inferred_mdp_grids:
        center_offset = 0.5 # Shifts points into center of cell.
        base_policy_grid = PlotHelp.PlotPolicy(maze, cmap, center_offset)
        for policy, use_print_keys, title, kernel_loc in zip(plot_policies, only_use_print_keys, titles,
                                                             kernel_locations):
            # Reorder policy dict for plotting.
            if use_print_keys: # VI and EM policies have DRA states in policy keys.
                list_of_tuples = [(key, policy[key]) for key in policy_keys_to_print]
            else: # Learned policy only has state numbers.
                order_of_keys = [key for key in range(grid_map.size)]
                list_of_tuples = [(key, policy[key]) for key in order_of_keys]
            policy = OrderedDict(list_of_tuples)
            fig = base_policy_grid.configurePlot(title, policy, action_list, use_print_keys, policy_keys_to_print,
                                                 decimals=2, kernel_locations=kernel_loc)
            plt.savefig('{}_solved.tif'.format(title), dpi=400, transparent=False)

        print '\n\nHEY! You! With the face! (computers don\'t have faces) Mazimize figure window to correctly show ' \
                'arrow/dot size ratio!\n'

    if plot_loaded_kernel or plot_new_kernel:
        if not perform_new_inference and plot_new_kernel:
            kernels = new_infer_mdp.kernels
        else:
            kernels = infer_mdp.kernels
        kernel_grid = PlotHelp.PlotKernel(maze, cmap, action_list, grid_map)
        kern_idx = 0
        title='Kernel Centered at {}.'.format(kern_idx)
        fig, ax = kernel_grid.configurePlot(title, kern_idx, kernels=kernels)

    if plot_loaded_phi or plot_new_phi:
        if not perform_new_inference and plot_new_phi:
            phi_at_state = new_infer_mdp.phi_at_state
        else:
            phi_at_state = infer_mdp.phi_at_state
        phi_grid = PlotHelp.PlotKernel(maze, cmap, action_list, grid_map)
        phi_idx = 0
        title=''
        for act in action_list:
            fig, ax = phi_grid.configurePlot(title, phi_idx, phi_at_state=phi_at_state, act=act)

    if plot_inference_statistics:
        infer_mdp.inferPolicy(method='historyMLE', histories=run_histories, do_print=False)
        mle_L1_norm = MDP.getPolicyL1Norm(reference_policy_vec, infer_mdp.getPolicyAsVec())
        print 'MLE L1 error is  {0:.3f}.'.format(mle_L1_norm)
        title = '' # Set to empty to format title in external program.
        PlotHelp.plotPolicyErrorVsNumberOfKernels(kernel_set_L1_err, num_kernels_in_set, title, mle_L1_norm)

    if any(plot_flags):
        plt.show()

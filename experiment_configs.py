#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

from NFA_DFA_Module.DFA import DRA
from MDP_EM.MDP_EM.MDP import MDP
from MDP_EM.MDP_EM.multi_agent_mdp import MultiAgentMDP
from MDP_EM.MDP_EM.product_mdp_x_dra import ProductMDPxDRA
import MDP_EM.MDP_EM.data_helper as DataHelper

import numpy as np
import time
from copy import deepcopy

def getActionProbabilityDictionary(dtype=np.float64):
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
                                  , dtype=dtype),
                'South': np.array([[0.0, 0.0, 0.8, 0.1, 0.1],
                                   [0.0, 0.0, 0.8, 0.1, 0.1],
                                   [0.8, 0.0, 0.0, 0.1, 0.1],
                                   [0.1, 0.0, 0.8, 0.0, 0.1],
                                   [0.1, 0.0, 0.8, 0.1, 0.0],
                                   [0.1, 0.0, 0.8, 0.0, 0.1],
                                   [0.1, 0.0, 0.8, 0.1, 0.0],
                                   [0.9, 0.0, 0.0, 0.0, 0.1],
                                   [0.9, 0.0, 0.0, 0.1, 0.0]]
                                  , dtype=dtype),
                'East': np.array([[0.0, 0.1, 0.1, 0.8, 0.0],
                                  [0.1, 0.0, 0.1, 0.8, 0.0],
                                  [0.1, 0.1, 0.0, 0.8, 0.0],
                                  [0.8, 0.1, 0.1, 0.0, 0.0],
                                  [0.0, 0.1, 0.1, 0.8, 0.0],
                                  [0.9, 0.0, 0.1, 0.0, 0.0],
                                  [0.1, 0.0, 0.1, 0.8, 0.0],
                                  [0.9, 0.1, 0.0, 0.0, 0.0],
                                  [0.1, 0.1, 0.0, 0.8, 0.0]]
                                 , dtype=dtype),
                'West': np.array([[0.0, 0.1, 0.1, 0.0, 0.8],
                                  [0.1, 0.0, 0.1, 0.0, 0.8],
                                  [0.1, 0.1, 0.0, 0.0, 0.8],
                                  [0.0, 0.1, 0.1, 0.0, 0.8],
                                  [0.8, 0.1, 0.1, 0.0, 0.0],
                                  [0.1, 0.0, 0.1, 0.0, 0.8],
                                  [0.9, 0.0, 0.1, 0.0, 0.0],
                                  [0.1, 0.1, 0.0, 0.0, 0.8],
                                  [0.9, 0.1, 0.0, 0.0, 0.0]]
                                 , dtype=dtype),
                'Empty': np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0]]
                                  , dtype=dtype),
                }
    return act_prob

def getDRAAvoidRedGetToGreen(alphabet_dict, save_dra_to_dot_file=False):
    """
    @param alphabet_dict At this point, this method expects the following keys to exist in the alphabet_dict: empty,
           red, green.
    """
    # Define a finite Deterministic Raban Automata to match the sketch on slide 7 of lecture 8. Note that state 'q2'
    # is is the red, 'sink' state.
    co_safe_dra = DRA(initial_state='q0', alphabet=alphabet_dict.items(), rabin_acc=[({'q1'},{})])
    # Self-loops = Empty transitions
    co_safe_dra.add_transition(alphabet_dict['empty'], 'q0', 'q0') # Initial state
    co_safe_dra.add_transition(alphabet_dict['empty'], 'q2', 'q2') # Losing sink.
    co_safe_dra.add_transition(alphabet_dict['empty'], 'q3', 'q3') # Winning sink.
    # Labeled transitions:
    # If the DRA reaches state 'q1' we win. Therefore I do not define a transition from 'q1' to 'q2'. Note that 'q2' and
    # 'q3' are a sink states due to the self loop.
    #
    # Also, I define a winning 'sink' state, 'q3'. I do this so that there is only one out-going transition from 'q1'
    # and it's taken only under the empty action. This action, is the winning action. This is a little bit of a hack,
    # but it was the way that I thought of to prevent the system from repeatedly taking actions that earned a reward.
    co_safe_dra.add_transition(alphabet_dict['green'], 'q0', 'q1')
    co_safe_dra.add_transition(alphabet_dict['green'], 'q1', 'q1') # State where winning action is available.
    co_safe_dra.add_transition(alphabet_dict['red'], 'q0', 'q2')
    co_safe_dra.add_transition(alphabet_dict['empty'], 'q1', 'q3')

    # Save DRA visualization file (can convert '.dot' file to PDF from terminal).
    if save_dra_to_dot_file:
        co_safe_dra.toDot('visitGreensAndNoRed.dot')
        pprint(vars(co_safe_dra))

    return co_safe_dra

def makeGridMDPxDRA(states, initial_state, action_set, alphabet_dict, labels, grid_map, gamma=0.9, act_prob=dict([]),
                    do_print=False, init_set=None,prob_dtype=np.float64):
    """
    @brief Configure the product MDP and DRA.

    @param act_prob Transition probabilities given _executed_ actions in location class in grid. If an empty-dict
           (default) is provided, this will be populated with getActionProbabilityDictionary().

    By default this will be constructed to satisfy the specification: visit the green cell and avoid all red cell.
    """
    if not any(act_prob):
        act_prob = getActionProbabilityDictionary(prob_dtype)

    if type(states[0]) is tuple and len(states[0]) > 1:
        grid_mdp = MultiAgentMDP(init=initial_state, action_dict=action_set, states=states,
                                 act_prob=deepcopy(act_prob), gamma=gamma, AP=alphabet_dict.items(), L=labels,
                                 grid_map=grid_map, init_set=init_set, prob_dtype=prob_dtype)
    else:
        grid_mdp = MDP(init=initial_state, action_list=action_set, states=states, act_prob=deepcopy(act_prob),
                       gamma=gamma, AP=alphabet_dict.items(), L=labels, grid_map=grid_map, init_set=init_set,
                       prob_dtype=prob_dtype)

    ##### Add DRA for co-safe spec #####
    co_safe_dra = getDRAAvoidRedGetToGreen(alphabet_dict)
    #### Create the Product MPDxDRA ####
    # Note that an MDPxDRA receives a binary reward upon completion of the specification so define the reward function
    # to re given when leaving the winning state on the winning action (from 'q1' to 'q3'). 'VI' implies this is to be
    # solved with Value Iteration.
    winning_reward = {'North': 0.0,
                      'South': 0.0,
                      'East': 0.0,
                      'West': 0.0,
                      'Empty': 1.0
        }
    VI_mdp = ProductMDPxDRA(grid_mdp, co_safe_dra, sink_action='Empty', sink_list=['q2', 'q3'],
                            losing_sink_label=alphabet_dict['red'], winning_reward=winning_reward,
                            prob_dtype=prob_dtype)

    # @TODO Prune unreachable states from MDP.

    # Create a dictionary of observable states for printing.
    policy_keys_to_print = deepcopy([(state[0], VI_mdp.dra.get_transition(VI_mdp.L[state], state[1])) for state in
                                     VI_mdp.states if 'q0' in state])
    VI_mdp.setObservableStates(observable_states=policy_keys_to_print)

    ##### SOLVE #####
    # To enable a solution of the MDP with multiple methods, copy the MDP, set the initial state likelihood
    # distributions and then solve the MDPs.
    EM_mdp = deepcopy(VI_mdp)
    EM_mdp.solve(do_print=do_print, method='expectationMaximization', write_video=False,
                 policy_keys_to_print=policy_keys_to_print)
    VI_mdp.solve(do_print=do_print, method='valueIteration', write_video=False,
                 policy_keys_to_print=policy_keys_to_print)

    # Compare the two solution methods.
    policy_difference, policy_KL_divergence  = MDP.comparePolicies(VI_mdp.policy, EM_mdp.policy, policy_keys_to_print,
                                                                   compare_to_decimals=3, do_print=do_print,
                                                                   compute_kl_divergence=True,
                                                                   reference_policy_has_augmented_states=True,
                                                                   compare_policy_has_augmented_states=True)

    return EM_mdp, VI_mdp, policy_keys_to_print, policy_difference

def makeMultiAgentGridMDPxDRA(states, initial_state, action_set, alphabet_dict, labels, grid_map, gamma=0.9,
                              act_prob=dict([]), do_print=False, init_set=None,prob_dtype=np.float64,
                              fixed_obstacle_labels=dict([]), use_mobile_kernels=False, gg_kernel_centers=[]):
    """
    @brief Configure the product MDP and DRA.

    @param act_prob Transition probabilities given _executed_ actions in location class in grid. If an empty-dict
           (default) is provided, this will be populated with getActionProbabilityDictionary().

    By default this will be constructed to satisfy the specification: visit the green cell and avoid all red cell.
    """
    if not any(act_prob):
        act_prob = getActionProbabilityDictionary(prob_dtype)

    if type(states[0]) is tuple and len(states[0]) > 1:
        grid_mdp = MultiAgentMDP(init=initial_state, action_dict=action_set, states=states,
                                 act_prob=deepcopy(act_prob), gamma=gamma, AP=alphabet_dict.items(), L=labels,
                                 grid_map=grid_map, init_set=init_set, prob_dtype=prob_dtype,
                                 fixed_obstacle_labels=fixed_obstacle_labels, use_mobile_kernels=use_mobile_kernels,
                                 ggk_centers=gg_kernel_centers)
    else:
        grid_mdp = MDP(init=initial_state, action_list=action_set, states=states, act_prob=deepcopy(act_prob),
                       gamma=gamma, AP=alphabet_dict.items(), L=labels, grid_map=grid_map, init_set=init_set,
                       prob_dtype=prob_dtype)

    ##### Add DRA for co-safe spec #####
    co_safe_dra = getDRAAvoidRedGetToGreen(alphabet_dict)
    #### Create the Product MPDxDRA ####
    # Note that an MDPxDRA receives a binary reward upon completion of the specification so define the reward function
    # to re given when leaving the winning state on the winning action (from 'q1' to 'q3'). 'VI' implies this is to be
    # solved with Value Iteration.
    winning_reward = {act: 0.0 for act in grid_mdp.action_list}
    winning_reward['0_Empty'] = 1.0
    skip_product_calcs = False
    if skip_product_calcs:
        sink_list = [state for state, label in labels.iteritems() if label is not alphabet_dict['empty']]
    else:
        sink_list = ['q2', 'q3']
    VI_mdp = ProductMDPxDRA(grid_mdp, co_safe_dra, sink_action='0_Empty', sink_list=sink_list,
                                 losing_sink_label=alphabet_dict['red'], winning_reward=winning_reward,
                                 prob_dtype=prob_dtype, skip_product_calcs=skip_product_calcs,
                                 winning_label=alphabet_dict['green'])

    # @TODO Prune unreachable states from MDP.

    # Create a dictionary of observable states for printing.
    if skip_product_calcs:
        policy_keys_to_print = VI_mdp.states
    else:
        policy_keys_to_print = deepcopy([(state[0], VI_mdp.dra.get_transition(VI_mdp.L[state], state[1])) for state in
                                         VI_mdp.states if 'q0' in state])
    VI_mdp.setObservableStates(observable_states=policy_keys_to_print)

    ##### SOLVE #####
    # To enable a solution of the MDP with multiple methods, copy the MDP, set the initial state likelihood
    # distributions and then solve the MDPs.
    VI_mdp.solve(do_print=do_print, method='valueIteration', write_video=False,
                 policy_keys_to_print=policy_keys_to_print)

    return VI_mdp, policy_keys_to_print


def rolloutInferSolve(arena_mdp, robot_idx, env_idx, num_batches=10, num_trajectories_per_batch=100,
        num_steps_per_traj=15, inference_method='gradientAscentGaussianTheta', infer_dtype=np.float64,
        num_theta_samples=2000, SGA_eps=0.00001, SGA_log_prob_thresh=np.log(0.8)):

    # Create a reference to the mdp used for inference.
    infer_mdp = arena_mdp.infer_env_mdp
    true_env_policy_vec = infer_mdp.getPolicyAsVec(policy_to_convert=arena_mdp.env_policy[env_idx])
    infer_mdp.theta = np.zeros(len(infer_mdp.phi))
    infer_mdp.theta_std_dev = np.ones(infer_mdp.theta.size)

    winning_reward = {act: 0.0 for act in arena_mdp.action_list}
    winning_reward['0_Empty'] = 1.0

    # Data types are constant for every batch.
    hist_dtype = DataHelper.getSmallestNumpyUnsignedIntType(arena_mdp.num_observable_states)
    observation_dtype  = DataHelper.getSmallestNumpyUnsignedIntType(arena_mdp.num_actions)

    # Create a dictionary of observable states for printing.
    policy_keys_to_print = deepcopy([(state[0], arena_mdp.dra.get_transition(arena_mdp.L[state], state[1])) for state in
                                     arena_mdp.states if 'q0' in state])

    # Variables for logging data
    inferred_policy = {}
    inferred_policy_L1_norm = {}
    inferred_policy_variance = {}


    for batch in range(num_batches):
        batch_start_time = time.time()
        ### Roll Out ###
        run_histories = np.zeros([num_trajectories_per_batch, num_steps_per_traj], dtype=hist_dtype)
        observed_action_indeces = np.empty([num_trajectories_per_batch, num_steps_per_traj], dtype=observation_dtype)
        for episode in xrange(num_trajectories_per_batch):
            # Create time-history for this episode.
            _, run_histories[episode, 0] = arena_mdp.resetState()
            for t_step in xrange(1, num_steps_per_traj):
                # Take step
                _, run_histories[episode, t_step] = arena_mdp.step()
                # Record observed action.
                prev_state_idx = run_histories[episode, t_step-1]
                prev_state = arena_mdp.observable_states[prev_state_idx]
                this_state_idx = run_histories[episode, t_step]
                this_state = arena_mdp.observable_states[this_state_idx]
                observed_action_indeces[episode, t_step] = infer_mdp.graph.getObservedAction(prev_state, this_state)

        ### Infer ###
        theta_vec = infer_mdp.inferPolicy(method=inference_method, histories=run_histories, do_print=False,
                                                       reference_policy_vec=true_env_policy_vec,
                                                       use_precomputed_phi=True, monte_carlo_size=num_theta_samples,
                                                       precomputed_observed_action_indeces=observed_action_indeces,
                                                       print_iterations=True, eps=0.00001, thresh_prob=0.3,
                                                       theta_0=infer_mdp.theta)
        # Print Inference error
        infered_policy_L1_norm_error = MDP.getPolicyL1Norm(true_env_policy_vec, infer_mdp.getPolicyAsVec())
        print('Batch {}: L1-norm from ref to inferred policy: {}.'.format(batch,infered_policy_L1_norm_error))
        print('L1-norm as a fraction of max error: {}.'.format(infered_policy_L1_norm_error/2/len(true_env_policy_vec)))

        # Go through and pop keys from policy_uncertainty into a dict built from policy_keys_to_print.
        arena_mdp.configureReward(winning_reward, bonus_reward_at_state=makeBonusReward(infer_mdp.policy_uncertainty))

        arena_mdp.solve(do_print=False, method='valueIteration', write_video=False,
                        policy_keys_to_print=policy_keys_to_print)
        batch_stop_time = time.time()
        print('Batch {} runtime {} sec.'.format(batch, batch_stop_time - batch_start_time))


def makeBonusReward(policy_uncertainty_dict):
    exploration_weight = 0.2
    bonus_reward_dict = dict.fromkeys(policy_uncertainty_dict)
    for state in policy_uncertainty_dict:
        bonus_reward_dict[state] = {}
        for act in policy_uncertainty_dict[state].keys():
            bonus_reward_dict[state][act] = exploration_weight * policy_uncertainty_dict[state][act]
    return bonus_reward_dict

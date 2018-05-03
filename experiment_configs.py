#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

from NFA_DFA_Module.DFA import DRA
from MDP_EM.MDP_EM.MDP import MDP
from MDP_EM.MDP_EM.inference_mdp import InferenceMDP
from MDP_EM.MDP_EM.multi_agent_mdp import MultiAgentMDP
from MDP_EM.MDP_EM.product_mdp_x_dra import ProductMDPxDRA
from MDP_EM.MDP_EM.feature_vector import FeatureVector
import MDP_EM.MDP_EM.data_helper as DataHelper
import MDP_EM.MDP_EM.plot_helper as PlotHelper

import numpy as np
import time
from copy import deepcopy
from collections import OrderedDict
from collections import deque
import matplotlib.pyplot as plt
import random

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
                    do_print=False, init_set=None,prob_dtype=np.float64, act_cost=0.0):
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
    winning_reward = {'North': act_cost,
                      'South': act_cost,
                      'East': act_cost,
                      'West': act_cost,
                      'Empty': 1.0
        }
    skip_product_calcs = False
    if skip_product_calcs:
        sink_list = [state for state, label in labels.iteritems() if label is not alphabet_dict['empty']]
    else:
        sink_list = ['q2', 'q3']
    VI_mdp = ProductMDPxDRA(grid_mdp, co_safe_dra, sink_action='Empty', sink_list=sink_list,
                            losing_sink_label=alphabet_dict['red'], winning_reward=winning_reward,
                            prob_dtype=prob_dtype, skip_product_calcs=skip_product_calcs,
                            winning_label=alphabet_dict['green'], act_cost=act_cost)

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
                              fixed_obstacle_labels=dict([]), use_mobile_kernels=False, gg_kernel_centers=[],
                              env_labels=None, print_demo_iterations=True, inference_temperature=1.0, act_cost=0.0,
                              use_em=True):
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
                                 ggk_centers=gg_kernel_centers, env_labels=env_labels,
                                 inference_temperature=inference_temperature)
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
    winning_reward = {act: act_cost for act in grid_mdp.action_list}
    winning_reward['0_Empty'] = 1.0
    skip_product_calcs = True
    if skip_product_calcs:
        sink_list = [state for state, label in labels.iteritems() if label is not alphabet_dict['empty']]
        if env_labels is not None:
            env_sink_list = [state for state, label in env_labels.iteritems() if label is not alphabet_dict['empty']]
        else:
            env_sink_list = []
    else:
        sink_list = ['q2', 'q3']
        env_sink_list = []
    demo_mdp = ProductMDPxDRA(grid_mdp, co_safe_dra, sink_action='0_Empty', sink_list=sink_list,
                              losing_sink_label=alphabet_dict['red'], winning_reward=winning_reward,
                              prob_dtype=prob_dtype, skip_product_calcs=skip_product_calcs,
                              winning_label=alphabet_dict['green'], env_sink_list=env_sink_list, act_cost=act_cost)

    # @TODO Prune unreachable states from MDP.

    # Create a dictionary of observable states for printing.
    if skip_product_calcs:
        policy_keys_to_print = demo_mdp.states
    else:
        policy_keys_to_print = deepcopy([(state[0], demo_mdp.dra.get_transition(demo_mdp.L[state], state[1])) for state
                                         in demo_mdp.states if 'q0' in state])
    demo_mdp.setObservableStates(observable_states=policy_keys_to_print)

    ##### SOLVE #####
    # To enable a solution of the MDP with multiple methods, copy the MDP, set the initial state likelihood
    # distributions and then solve the MDPs.
    if use_em:
        method = 'expectationMaximization'
    else:
        method = 'valueIteration'
    demo_mdp.solve(do_print=do_print, method=method, horizon_length=15, num_iters=100, do_incremental_e_step=True,
                   policy_keys_to_print=policy_keys_to_print, print_iterations=print_demo_iterations)

    return demo_mdp, policy_keys_to_print

def rolloutInferSolve(arena_mdp, robot_idx, env_idx, num_batches=10, num_trajectories_per_batch=100,
                      num_steps_per_traj=15, inference_method='gradientAscentGaussianTheta', infer_dtype=np.float64,
                      num_theta_samples=1000, robot_goal_states=None, act_cost=0.0, use_active_inference=True,
                      true_optimal_VI_policy=None):

    print "Using {} inference.".format("active" if use_active_inference else "passive")

    robot_action_list = arena_mdp.executable_action_dict[0]
    env_action_list = arena_mdp.executable_action_dict[1]

    # Create a reference to the mdp used for inference.
    # Double check that mdp has correct action list to build the vector in the right order.
    infer_mdp = arena_mdp.infer_env_mdp
    true_env_policy_vec = infer_mdp.getPolicyAsVec(policy_to_convert=arena_mdp.env_policy[env_idx],
                                                   action_list=env_action_list)
    infer_mdp.theta = np.zeros(len(infer_mdp.phi))
    infer_mdp.theta_std_dev = np.ones(infer_mdp.theta.size)

    winning_reward = {act: act_cost for act in arena_mdp.action_list}
    winning_reward['0_Empty'] = 1.0

    # Data types are constant for every batch.
    hist_dtype = DataHelper.getSmallestNumpyUnsignedIntType(arena_mdp.num_observable_states)
    observation_dtype  = DataHelper.getSmallestNumpyUnsignedIntType(arena_mdp.num_actions)

    # Create a dictionary of observable states for printing.
    policy_keys_to_print = deepcopy([(state[0], arena_mdp.dra.get_transition(arena_mdp.L[state], state[1])) for state in
                                     arena_mdp.states if 'q0' in state])

    # Variables for logging data
    initial_policy_guess = arena_mdp.infer_env_mdp.getPolicyAsVec()
    L1_norm_of_initial_policy_guess = MDP.getPolicyL1Norm(true_env_policy_vec, initial_policy_guess)
    inferred_policy = [initial_policy_guess]
    recorded_inferred_policy_L1_norms = [L1_norm_of_initial_policy_guess]
    inferred_policy_variance = [np.sum(np.ones(infer_mdp.theta.size))]
    reward_counts = []
    bonus_reward_mags = []
    robot_policy_err_to_opt = [MDP.getPolicyL1Norm(true_optimal_VI_policy, arena_mdp.getPolicyAsVec())]

    theta_std_dev_min = 0.4

    # Preallocate an array to hold trajectory rollouts, a.k.a demonstrations/histories.
    run_histories = np.zeros([num_trajectories_per_batch * num_batches, num_steps_per_traj], dtype=hist_dtype)
    executed_robot_actions = np.zeros([num_trajectories_per_batch * num_batches, num_steps_per_traj], dtype=hist_dtype)
    observed_action_indices = np.empty([num_trajectories_per_batch * num_batches, num_steps_per_traj],
                                       dtype=observation_dtype)
    observed_action_probs = np.empty([num_trajectories_per_batch * num_batches, num_steps_per_traj], dtype=infer_dtype)

    for batch in range(num_batches):
        batch_start_time = time.time()
        batch_idx = batch * num_trajectories_per_batch

        ## Roll Out ###
        for episode in xrange(num_trajectories_per_batch):
            # Create time-history for this episode.
            hist_idx = (batch_idx) + episode
            _, run_histories[hist_idx, 0] = arena_mdp.resetState()
            for t_step in xrange(1, num_steps_per_traj):
                # Take step
                _, run_histories[hist_idx, t_step], executed_robot_action = arena_mdp.step()
                executed_robot_actions[hist_idx, t_step] = robot_action_list.index(executed_robot_action)
                # Record observed action.
                prev_state_idx = run_histories[hist_idx, t_step-1]
                prev_state = arena_mdp.observable_states[prev_state_idx]
                this_state_idx = run_histories[hist_idx, t_step]
                this_state = arena_mdp.observable_states[this_state_idx]
                observed_action_indices[hist_idx, t_step] = infer_mdp.graph.getObservedAction(prev_state, this_state)
                robot_act = robot_action_list[executed_robot_actions[hist_idx, t_step]]
                env_act = env_action_list[observed_action_indices[hist_idx, t_step]]
                observed_action_probs[hist_idx, t_step] = arena_mdp.P(prev_state, robot_act, env_act, this_state)

        if robot_goal_states is not None:
            reward_frac, reward_count = DataHelper.printHistoryAnalysis(run_histories[:hist_idx + 1], arena_mdp.states,
                                                                        arena_mdp.L, None, robot_goal_states)
            reward_counts.append(reward_count)

        DataHelper.printStateHistories(run_histories[batch_idx:hist_idx + 1], arena_mdp.observable_states)
        nominal_log_prob_data = np.log(observed_action_probs[:hist_idx + 1, 1:]).sum()

        ### Infer ###
        theta_std_dev_0 = infer_mdp.theta_std_dev

        # Since the gradient variance is proportional to the size of the demonstration, we'll set the gradient ascent
        # step size to be inversly proportional to the negative of the log-probablilty of the observed data given the
        # MAP estimate of the observed action outcomes.
        if nominal_log_prob_data != 0.0:
            eps = 0.1 / (-nominal_log_prob_data)
        else:
            eps = 0.1 / (hist_idx + 1)
        theta_vec = infer_mdp.inferPolicy(method=inference_method, histories=run_histories[:hist_idx + 1],
                                          do_print=False, theta_std_dev_0=infer_mdp.theta_std_dev,
                                          theta_0=infer_mdp.theta, reference_policy_vec=true_env_policy_vec,
                                          monte_carlo_size=num_theta_samples, print_iterations=False, eps=eps,
                                          velocity_memory=0.2, theta_std_dev_min=theta_std_dev_min,
                                          theta_std_dev_max=np.inf, nominal_log_prob_data=nominal_log_prob_data,
                                          moving_avg_min_slope=0.001, moving_average_buffer_length=60, do_plot=False,
                                          precomputed_observed_action_indices=observed_action_indices[:hist_idx + 1],
                                          min_uncertainty=1.0)

        # Print Inference error
        # Check getPolicyAsVec for this MDP!
        inferred_policy_L1_norm_error = MDP.getPolicyL1Norm(true_env_policy_vec, infer_mdp.getPolicyAsVec())
        print('Batch {}: L1-norm from ref to inferred policy: {}.'.format(batch, inferred_policy_L1_norm_error))
        print('L1-norm as a fraction of max error: {}.'.format(inferred_policy_L1_norm_error/2/infer_mdp.num_states))
        recorded_inferred_policy_L1_norms.append(inferred_policy_L1_norm_error)
        inferred_policy_variance.append(np.sum(np.power(infer_mdp.theta_std_dev, 2)))

        if use_active_inference:
            # Go through and pop keys from policy_uncertainty into a dict built from policy_keys_to_print.
            bonus_reward_dict = makeBonusReward(infer_mdp.policy_uncertainty)
            arena_mdp.configureReward(winning_reward, bonus_reward_at_state=bonus_reward_dict, act_cost=act_cost)
            bonus_reward_vec = arena_mdp.getPolicyAsVec(policy_keys_to_use=bonus_reward_dict.keys(),
                                                        policy_to_convert=bonus_reward_dict,
                                                        action_list=env_action_list)
            bonus_reward_mag = np.sum(bonus_reward_vec)
        else:
            bonus_reward_mag = 0
        bonus_reward_mags.append(bonus_reward_mag)
        # Need to reset the policy to something _very_ sub-optimal for EM.
        arena_mdp.makeUniformPolicy()
        arena_mdp.solve(do_print=False, method='expectationMaximization', print_iterations=False,
                        horizon_length=15, num_iters=100, do_incremental_e_step=True)
        robot_policy_err_to_opt.append(MDP.getPolicyL1Norm(true_optimal_VI_policy, arena_mdp.getPolicyAsVec()))
        batch_stop_time = time.time()
        print('Batch {} runtime {} sec.'.format(batch, batch_stop_time - batch_start_time))

    return (recorded_inferred_policy_L1_norms, reward_counts, inferred_policy_variance, bonus_reward_mags,
            robot_policy_err_to_opt)

def rolloutInferSingleAgent(env_mdp, infer_mdp, num_batches=10, num_trajectories_per_batch=100, num_steps_per_traj=15,
                            inference_method='gradientAscentGaussianTheta', infer_dtype=np.float64,
                            num_theta_samples=1000, robot_goal_states=None, use_active_inference=True):

    print "Using {} inference.".format("active" if use_active_inference else "passive")

    # Create a dictionary of observable states for printing, this removes the 'dra' states, so we can then turn it into
    # a vector for numerical comparison (infinite-norm).
    policy_keys_to_print = deepcopy([(state[0], env_mdp.dra.get_transition(env_mdp.L[state], state[1])) for state in
                                     env_mdp.states if 'q0' in state])
    true_env_policy_vec = env_mdp.getPolicyAsVec(policy_keys_to_print)

    # Initial inference params.
    infer_mdp.theta = np.zeros(len(infer_mdp.phi))
    infer_mdp.theta_std_dev = np.ones(infer_mdp.theta.size)

    # Data types are constant for every batch.
    hist_dtype = DataHelper.getSmallestNumpyUnsignedIntType(env_mdp.num_observable_states)
    observation_dtype = DataHelper.getSmallestNumpyUnsignedIntType(env_mdp.num_actions)


    # Variables for logging data
    initial_policy_guess = infer_mdp.getPolicyAsVec()
    L1_norm_of_initial_policy_guess = MDP.getPolicyL1Norm(true_env_policy_vec, initial_policy_guess)
    inferred_policy = [initial_policy_guess]
    recorded_inferred_policy_L1_norms = [L1_norm_of_initial_policy_guess]
    inferred_policy_variance = [np.sum(np.ones(infer_mdp.theta.size))]
    reward_fractions = []

    theta_std_dev_min = 0.1

    # Ensure initial state for batch 0 will be uniformly, randomly selected.
    env_mdp.init_set = policy_keys_to_print
    env_mdp.setInitialProbDist(env_mdp.init_set)

    # Preallocate an array to hold trajectory rollouts, a.k.a demonstrations/histories.
    run_histories = np.zeros([num_trajectories_per_batch * num_batches, num_steps_per_traj], dtype=hist_dtype)
    observed_action_indices = np.empty([num_trajectories_per_batch * num_batches, num_steps_per_traj], dtype=observation_dtype)
    observed_action_probs = np.empty([num_trajectories_per_batch * num_batches, num_steps_per_traj], dtype=infer_dtype)

    for batch in range(num_batches):
        if use_active_inference and batch > 0 and not any(np.isnan(active_initial_dist)) and (batch % 2):
            # Select the state with the highest uncertainty to be the initial state for all trajectories. Another way to
            # do this would be to have a prioritiezed weighting of states based off of their variance.
            env_mdp.setInitialProbDist(env_mdp.init_set, init_prob=active_initial_dist)
        else:
            env_mdp.setInitialProbDist(env_mdp.init_set)

        batch_start_time = time.time()
        batch_idx = batch * num_trajectories_per_batch
        ### Roll Out ###
        for episode in xrange(num_trajectories_per_batch):
            # Create time-history for this episode.
            hist_idx = (batch_idx) + episode
            _, run_histories[hist_idx, 0] = env_mdp.resetState()
            for t_step in xrange(1, num_steps_per_traj):
                # Take step
                _, run_histories[hist_idx, t_step] = env_mdp.step()
                # Record observed action.
                prev_state_idx = run_histories[hist_idx, t_step-1]
                prev_state = env_mdp.observable_states[prev_state_idx]
                this_state_idx = run_histories[hist_idx, t_step]
                this_state = env_mdp.observable_states[this_state_idx]
                obs_act_idx = infer_mdp.graph.getObservedAction(prev_state, this_state)
                observed_action_indices[hist_idx, t_step] = obs_act_idx
                if 0 == infer_mdp.P(prev_state[0], infer_mdp.action_list[obs_act_idx], this_state[0]):
                    import pdb; pdb.set_trace()
                observed_action_probs[hist_idx, t_step] = infer_mdp.P(prev_state[0], infer_mdp.action_list[obs_act_idx],
                                                                     this_state[0])

        DataHelper.printStateHistories(run_histories[batch_idx:hist_idx + 1], env_mdp.observable_states)
        nominal_log_prob_data = np.log(observed_action_probs[:hist_idx + 1, 1:]).sum()
        print "Nomainal log prob data: {}".format(nominal_log_prob_data)

        ### Infer ###
        # Since the gradient variance is proportional to the size of the demonstration, we'll set the gradient ascent
        # step size to be inversly proportional to the negative of the log-probablilty of the observed data given the
        # MAP estimate of the observed action outcomes.
        if nominal_log_prob_data != 0.0:
            eps = 0.01 / (-nominal_log_prob_data)
        else:
            eps = 0.01 / (hist_idx + 1)
        infer_mdp.inferPolicy(method=inference_method, histories=run_histories[:hist_idx + 1], do_print=False,
                              theta_std_dev_0=infer_mdp.theta_std_dev, theta_0=infer_mdp.theta,
                              reference_policy_vec=true_env_policy_vec, monte_carlo_size=num_theta_samples,
                              print_iterations=False, eps=eps, velocity_memory=0.0, theta_std_dev_min=theta_std_dev_min,
                              theta_std_dev_max=np.inf, nominal_log_prob_data=nominal_log_prob_data,
                              moving_avg_min_slope=0.001, moving_average_buffer_length=60, do_plot=False,
                              precomputed_observed_action_indices=observed_action_indices[:hist_idx + 1],
                              min_uncertainty=theta_std_dev_min)

        # Print Inference error
        # Check getPolicyAsVec for this MDP!
        inferred_policy_L1_norm_error = MDP.getPolicyL1Norm(true_env_policy_vec, infer_mdp.getPolicyAsVec())
        print('Batch {}: L1-norm from ref to inferred policy: {}.'.format(batch, inferred_policy_L1_norm_error))
        print('L1-norm as a fraction of max error: {}.'.format(inferred_policy_L1_norm_error/2/infer_mdp.num_states))
        recorded_inferred_policy_L1_norms.append(inferred_policy_L1_norm_error)
        inferred_policy_variance.append(np.sum(np.power(infer_mdp.theta_std_dev, 2)))

        if use_active_inference:
            active_initial_dist = \
                np.sum(infer_mdp.policy_uncertainty_as_vec.reshape(infer_mdp.num_states, infer_mdp.num_actions), axis=1)
            active_initial_dist /= sum(active_initial_dist)
            start_at_max_unc = np.zeros(active_initial_dist.shape)
            start_at_max_unc[np.argmax(active_initial_dist)] = 1.0
        batch_stop_time = time.time()
        print('Batch {} runtime {} sec.'.format(batch, batch_stop_time - batch_start_time))

    return recorded_inferred_policy_L1_norms, inferred_policy_variance

def makeBonusReward(policy_uncertainty_dict):
    exploration_weight = 0.2
    bonus_reward_dict = dict.fromkeys(policy_uncertainty_dict)
    for state in policy_uncertainty_dict:
        bonus_reward_dict[state] = {}
        for act in policy_uncertainty_dict[state].keys():
            bonus_reward_dict[state][act] = exploration_weight * policy_uncertainty_dict[state][act]
    return bonus_reward_dict

def augmentSoftmaxPolicy(original_theta, original_phi_at_state, additional_theta, additional_phi, trans_function,
                         policy_keys_to_use, action_list, temperature=0.5, state_idx_to_use=0):
    """
    @brief Augment an original policy with additional parameters and features.

    @param original_theta A vector of weights
    @param original_phi_at_state A reference to @ref InferenceMDP.phi_at_state and this is expected to accept state
    inputs for a single-agent s
    @param additional_theta A vector of weights
    @param additional_phi_at_state An instance of @ref FeatureVector
    @param policy_keys_to_use A set or list of states to use as policy keys. States are expected to be a tuple, single
           agent states should be listed as [(state_1,), (state_2,), ...] and multi-agent states as [(agent_1_state_1,
           agent_2_state_1), (agent_1_state_2, agent_2_state_2) ...].
    @param temperature The policy temperatuer to use for synthesis
    @param state_idx_to_use This is used to index into the states in policy_keys_to_use.

    @note
    It probably makes sense to pass in the original @ref FeatureVector for original_phi_at_state instead of the
    precomputed dictionary, see InferenceMDP.precomputePhiAtState().

    @return A synthesized softmax policy built as exp(Q'/temp)/sum_over_actions(exp(Q'/temp)) where
    Q'(s,a) = inner(original_thata, original_phi(state,action)) + inner(additional_theta, additional_phi)
    """
    new_policy_dict = dict.fromkeys(policy_keys_to_use)
    for state in policy_keys_to_use:
        new_policy_dict[state] = {}
        Q_at_state = {act: (np.dot(original_theta, original_phi_at_state[(state[state_idx_to_use],)][act])
                            + np.dot(additional_theta, additional_phi(state, act))) / temperature
                      for act in action_list}
        new_policy_dict[state] = InferenceMDP.evalGibbsPolicy(Q_at_state, temperature)
    return new_policy_dict

def convertSingleAgentEnvPolicyToMultiAgent(multi_agent_mdp, joint_state_labels, state_env_idx=1, file_with_policy=None,
                                            new_kernel_weight=1.0, new_phi_sigma=1.0, plot_policies=True,
                                            alphabet_dict=None, fixed_obstacle_labels=None):
    """
    @brief Converts a single-agent policy dictionary to a multi-agent policy dictionary with the added repulsive factor.

    This converts a loaded environmental policy to a multi-agent policy dictionary and includes a repulsive factor based
    on the robot position. The loaded mdp must have a `theta` and `phi` property that can be used to build the soft-max
    policy as a function of the linear combination of these vectors. This method adds a mobile kernel at the robot
    location to augment the environments Q-function. Then the policy is recalculated.

    @param file_with_policy The file-path to load that contains a singe agent policy.
    @param new_kernel_weight @todo
    @param new_kernel_sigma @todo
    @param alphabet_dict Required if plot_policys is True
    """
    if file_with_policy is None:
        file_with_policy =  'robot_mdps_180412_2025_HIST_500eps10steps_180412_2026_Policy_180412_2026'

    print 'Next File loaded is for building environmental policy:'
    (single_agent_mdp, pickled_inference_file) = DataHelper.loadPickledPolicyInferenceMDP(file_with_policy)

    # Build a feature vector that only has a mobile kernel on the robot location.
    copied_single_agent_mdp = deepcopy(single_agent_mdp)
    joint_grid_states = multi_agent_mdp.env_policy[1].keys()
    env_action_list = copied_single_agent_mdp.action_list
    trans_func = copied_single_agent_mdp.T
    repulsive_feature_vector = FeatureVector(env_action_list, trans_func, copied_single_agent_mdp.graph,
                                             ggk_centers=[0], std_devs=[3.0], ggk_mobile_indices=[0],
                                             state_list=joint_grid_states, state_idx_to_infer=1,
                                             mobile_kernel_state_idx=0)
    #                           0, N, S, E, W
    repulsive_theta = np.array([0.,10,10.,10.,10.])


    theta_lengt = copied_single_agent_mdp.theta
    # Linearly combine old Q-function with repulsive Q-function.
    new_env_policy = augmentSoftmaxPolicy(copied_single_agent_mdp.theta, copied_single_agent_mdp.phi_at_state,
                                          repulsive_theta, repulstive_feature_vector,
                                          trans_function=copied_single_agent_mdp.T,
                                          policy_keys_to_use=joint_grid_states, action_list=env_action_list,
                                          temperature=0.1)
    new_env_policy = MDP.updatePolicyActionKeys(new_env_policy, env_action_list,
                                                multi_agent_mdp.executable_action_dict[state_env_idx])
    multi_agent_mdp.env_policy[1] = new_env_policy


    # The 'q0' below is a hack to make sure that the environment's goal cell is only highlighted once and therefore
    # the colormap is handed the correct color range.
    env_goal_cells = [cell[0] for cell, label in single_agent_mdp.L.iteritems() if label==alphabet_dict['green']]
    joint_env_goal_states = [((robot_cell, env_goal),'q0') for robot_cell in single_agent_mdp.grid_cell_vec for env_goal
                             in env_goal_cells]

    if plot_policies:

        single_agent_maze, single_agent_cmap = PlotHelper.PlotGrid.buildGridPlotArgs(single_agent_mdp.grid_map,
                                                                                     single_agent_mdp.L,
                                                                                     alphabet_dict)
        multi_agent_maze, multi_agent_cmap = PlotHelper.PlotGrid.buildGridPlotArgs(multi_agent_mdp.grid_map,
            multi_agent_mdp.L, alphabet_dict, num_agents=2, agent_idx=1, fixed_obstacle_labels=fixed_obstacle_labels,
            goal_states=joint_env_goal_states, labels_have_dra_states=True)

        center_offset = 0.5 # Shifts points into center of cell.
        single_agent_policy_grid = PlotHelper.PlotPolicy(single_agent_maze, single_agent_cmap, center_offset)
        multi_agent_policy_grid = PlotHelper.PlotPolicy(multi_agent_maze, multi_agent_cmap, center_offset)

         # Configure Single agent policy plot
        order_of_keys = [key for key in single_agent_mdp.states]
        list_of_tuples = [(key, single_agent_mdp.policy[key]) for key in order_of_keys]
        single_agent_ordered_policy = OrderedDict(list_of_tuples)
        fig = single_agent_policy_grid.configurePlot('Original Policy', single_agent_ordered_policy,
                single_agent_mdp.action_list, decimals=2)

        # Configure Multi agent policy plots. We'll create a plot for every robot location in the joint space.
        multi_agent_policy_key_groups = {robot_cell:
            [(robot_cell, env_cell) for env_cell in multi_agent_mdp.grid_cell_vec] for robot_cell in
            multi_agent_mdp.grid_cell_vec}

        # Reorder policy dict for plotting.
        order_of_keys = [key[0] for key in multi_agent_mdp.observable_states]
        list_of_tuples = [(key, multi_agent_mdp.env_policy[1][key]) for key in order_of_keys]
        policy = OrderedDict(list_of_tuples)

        fixed_idx = 0 # Robot pose is fixed in plots below.
        robot_idx = 0
        env_idx = 1
        for pose in multi_agent_policy_key_groups.keys():

            # Get policy at desired states to plot.
            list_of_tuples = [(key, policy[key]) for key in multi_agent_policy_key_groups[pose]]
            policy_to_plot = OrderedDict(list_of_tuples)

            # Update the grid colors, assuming the environment is in a fixed locaiton.
            this_maze = deepcopy(multi_agent_maze)
            grid_row, grid_col = np.where(multi_agent_mdp.grid_map==pose)
            this_maze[grid_row, grid_col] = multi_agent_cmap.colors.index('blue')
            multi_agent_policy_grid.updateCellColors(maze_cells=this_maze)

            act_list = multi_agent_mdp.executable_action_dict[state_env_idx]
            fig = multi_agent_policy_grid.configurePlot('Joint Policy - Robot in cell {}'.format(pose), policy_to_plot,
                                                 act_list, use_print_keys=True,
                                                 policy_keys_to_print=multi_agent_policy_key_groups[pose], decimals=2,
                                                 stay_action=act_list[0])
        plt.draw()


def rolloutInferResample(env_mdp, infer_mdp, initial_traj_count=20, initial_traj_length=10,
                                               second_traj_count=20, second_traj_length=2,
                                               inference_method='gradientAscentGaussianTheta', infer_dtype=np.float64,
                                               num_theta_samples=1000, robot_goal_states=None,
                                               use_active_inference=True):

    print "Using {} inference.".format("active" if use_active_inference else "passive")

    # Create a dictionary of observable states for printing, this removes the 'dra' states, so we can then turn it into
    # a vector for numerical comparison (infinite-norm).
    policy_keys_to_print = deepcopy([(state[0], env_mdp.dra.get_transition(env_mdp.L[state], state[1])) for state in
                                     env_mdp.states if 'q0' in state])
    true_env_policy_vec = env_mdp.getPolicyAsVec(policy_keys_to_print)

    # Initial inference params.
    infer_mdp.theta = np.zeros(len(infer_mdp.phi))
    infer_mdp.theta_std_dev = np.ones(infer_mdp.theta.size)

    # Data types are constant for every batch.
    hist_dtype = DataHelper.getSmallestNumpyUnsignedIntType(env_mdp.num_observable_states)
    observation_dtype = DataHelper.getSmallestNumpyUnsignedIntType(env_mdp.num_actions)


    # Variables for logging data
    initial_policy_guess = infer_mdp.getPolicyAsVec()
    L1_norm_of_initial_policy_guess = MDP.getPolicyL1Norm(true_env_policy_vec, initial_policy_guess)
    recorded_inferred_policy_L1_norms = [L1_norm_of_initial_policy_guess]
    inferred_policy_variance = [np.sum(np.ones(infer_mdp.theta.size))]

    theta_std_dev_min = 0.1

    # Ensure initial state for batch 0 will be uniformly, randomly selected.
    env_mdp.init_set = policy_keys_to_print
    env_mdp.setInitialProbDist(env_mdp.init_set)

    # Preallocate an array to hold trajectory rollouts, a.k.a demonstrations/histories.
    run_histories = np.zeros([initial_traj_count, initial_traj_length], dtype=hist_dtype)
    observed_action_indices = np.empty([initial_traj_count, initial_traj_length], dtype=observation_dtype)
    observed_action_probs = np.empty([initial_traj_count, initial_traj_length], dtype=infer_dtype)
    # Additional samples
    additional_samples = np.zeros([second_traj_count, second_traj_length], dtype=hist_dtype)
    additional_observed_action_indices = np.empty([second_traj_count, second_traj_length], dtype=observation_dtype)
    additional_observed_action_probs = np.empty([second_traj_count, second_traj_length], dtype=infer_dtype)

    ### Roll Out ###
    for episode in xrange(initial_traj_count):
        # Create time-history for this episode.
        hist_idx = episode
        _, run_histories[hist_idx, 0] = env_mdp.resetState()
        for t_step in xrange(1, initial_traj_length):
            # Take step
            _, run_histories[hist_idx, t_step] = env_mdp.step()
            # Record observed action.
            prev_state_idx = run_histories[hist_idx, t_step-1]
            prev_state = env_mdp.observable_states[prev_state_idx]
            this_state_idx = run_histories[hist_idx, t_step]
            this_state = env_mdp.observable_states[this_state_idx]
            obs_act_idx = infer_mdp.graph.getObservedAction(prev_state, this_state)
            observed_action_indices[hist_idx, t_step] = obs_act_idx
            if 0 == infer_mdp.P(prev_state[0], infer_mdp.action_list[obs_act_idx], this_state[0]):
                import pdb; pdb.set_trace()
            observed_action_probs[hist_idx, t_step] = infer_mdp.P(prev_state[0], infer_mdp.action_list[obs_act_idx],
                                                                 this_state[0])

    DataHelper.printStateHistories(run_histories, env_mdp.observable_states)
    nominal_log_prob_data = np.log(observed_action_probs[:, 1:]).sum()

    ### Infer ###
    # Since the gradient variance is proportional to the size of the demonstration, we'll set the gradient ascent
    # step size to be inversly proportional to the negative of the log-probablilty of the observed data given the
    # MAP estimate of the observed action outcomes.
    if nominal_log_prob_data != 0.0:
        eps = 0.01 / (-nominal_log_prob_data)
    else:
        eps = 0.01 / (hist_idx + 1)
    infer_mdp.inferPolicy(method=inference_method, histories=run_histories, do_print=False,
                          theta_std_dev_0=infer_mdp.theta_std_dev, theta_0=infer_mdp.theta,
                          reference_policy_vec=true_env_policy_vec, monte_carlo_size=num_theta_samples,
                          print_iterations=False, eps=eps, velocity_memory=0.0, theta_std_dev_min=theta_std_dev_min,
                          theta_std_dev_max=np.inf, nominal_log_prob_data=nominal_log_prob_data,
                          moving_avg_min_slope=0.001, moving_average_buffer_length=60, do_plot=False,
                          precomputed_observed_action_indices=observed_action_indices[:hist_idx + 1],
                          min_uncertainty=theta_std_dev_min)

    # Print Inference error
    # Check getPolicyAsVec for this MDP!
    inferred_policy_L1_norm_error = MDP.getPolicyL1Norm(true_env_policy_vec, infer_mdp.getPolicyAsVec())
    print('First Try: L1-norm as a fraction of max error: {}.'.format(inferred_policy_L1_norm_error/2/infer_mdp.num_states))
    recorded_inferred_policy_L1_norms.append(inferred_policy_L1_norm_error)
    inferred_policy_variance.append(np.sum(np.power(infer_mdp.theta_std_dev, 2)))

########################################################################################################################
    ## PASSIVE RESAMPLING
    # Using the same initial distribution (uniform) sample more data.
    passive_policy_L1_norms = deepcopy(recorded_inferred_policy_L1_norms)
    passive_policy_variance = deepcopy(inferred_policy_variance)
    passive_infer_mdp = deepcopy(infer_mdp)
    passive_additional_samples = deepcopy(additional_samples)
    passive_additional_observed_action_indices = deepcopy(additional_observed_action_indices)
    passive_additional_observed_action_probs = deepcopy(additional_observed_action_probs)

    # Resamples
    for episode in xrange(second_traj_count):
        # Create time-history for this episode.
        hist_idx = episode
        _, passive_additional_samples[hist_idx, 0] = env_mdp.resetState()
        for t_step in xrange(1, second_traj_length):
            # Take step
            _, passive_additional_samples[hist_idx, t_step] = env_mdp.step()
            # Record observed action.
            prev_state_idx = passive_additional_samples[hist_idx, t_step-1]
            prev_state = env_mdp.observable_states[prev_state_idx]
            this_state_idx = passive_additional_samples[hist_idx, t_step]
            this_state = env_mdp.observable_states[this_state_idx]
            obs_act_idx = passive_infer_mdp.graph.getObservedAction(prev_state, this_state)
            passive_additional_observed_action_indices[hist_idx, t_step] = obs_act_idx
            if 0 == passive_infer_mdp.P(prev_state[0], passive_infer_mdp.action_list[obs_act_idx], this_state[0]):
                import pdb; pdb.set_trace()
            passive_additional_observed_action_probs[hist_idx, t_step] = passive_infer_mdp.P(prev_state[0], passive_infer_mdp.action_list[obs_act_idx],
                                                                 this_state[0])

    DataHelper.printStateHistories(passive_additional_samples, env_mdp.observable_states)
    passive_nominal_log_prob_data = (np.log(passive_additional_observed_action_probs[:, 1:]).sum() +
                                     nominal_log_prob_data)

    if nominal_log_prob_data != 0.0:
        eps = 0.01 / (-nominal_log_prob_data)
    else:
        eps = 0.01 / (hist_idx + 1)
    passive_infer_mdp.inferPolicy(method=inference_method, histories=run_histories, do_print=False,
                          theta_std_dev_0=passive_infer_mdp.theta_std_dev, theta_0=passive_infer_mdp.theta,
                          reference_policy_vec=true_env_policy_vec, monte_carlo_size=num_theta_samples,
                          print_iterations=False, eps=eps, velocity_memory=0.0, theta_std_dev_min=theta_std_dev_min,
                          theta_std_dev_max=np.inf, nominal_log_prob_data=nominal_log_prob_data,
                          moving_avg_min_slope=0.001, moving_average_buffer_length=60, do_plot=False,
                          precomputed_observed_action_indices=observed_action_indices,
                          additional_precomputed_observed_action_indices=passive_additional_observed_action_indices,
                          additional_samples=passive_additional_samples,
                          min_uncertainty=theta_std_dev_min)
    inferred_policy_L1_norm_error = MDP.getPolicyL1Norm(true_env_policy_vec, passive_infer_mdp.getPolicyAsVec())
    print('PASSIVE with resamples: L1-norm as a fraction of max error: {}.'.format(inferred_policy_L1_norm_error/2/infer_mdp.num_states))
    passive_policy_L1_norms.append(inferred_policy_L1_norm_error)
    passive_policy_variance.append(np.sum(np.power(passive_infer_mdp.theta_std_dev, 2)))

########################################################################################################################
    ## ACTIVE RESAMPLING
    # Using the same initial distribution (uniform) sample more data.
    active_policy_L1_norms = deepcopy(recorded_inferred_policy_L1_norms)
    active_policy_variance = deepcopy(inferred_policy_variance)
    active_infer_mdp = deepcopy(infer_mdp)
    active_additional_samples = deepcopy(additional_samples)
    active_additional_observed_action_indices = deepcopy(additional_observed_action_indices)
    active_additional_observed_action_probs = deepcopy(additional_observed_action_probs)

    # Reconfigure initial ditribution
    active_initial_dist = \
        np.sum(active_infer_mdp.policy_uncertainty_as_vec.reshape(active_infer_mdp.num_states, active_infer_mdp.num_actions), axis=1)
    active_initial_dist /= sum(active_initial_dist)
    if not any(np.isnan(active_initial_dist)):
        env_mdp.setInitialProbDist(env_mdp.init_set, init_prob=active_initial_dist)

    # Resamples
    for episode in xrange(second_traj_count):
        # Create time-history for this episode.
        hist_idx = episode
        _, active_additional_samples[hist_idx, 0] = env_mdp.resetState()
        for t_step in xrange(1, second_traj_length):
            # Take step
            _, active_additional_samples[hist_idx, t_step] = env_mdp.step()
            # Record observed action.
            prev_state_idx = active_additional_samples[hist_idx, t_step-1]
            prev_state = env_mdp.observable_states[prev_state_idx]
            this_state_idx = active_additional_samples[hist_idx, t_step]
            this_state = env_mdp.observable_states[this_state_idx]
            obs_act_idx = active_infer_mdp.graph.getObservedAction(prev_state, this_state)
            active_additional_observed_action_indices[hist_idx, t_step] = obs_act_idx
            if 0 == active_infer_mdp.P(prev_state[0], active_infer_mdp.action_list[obs_act_idx], this_state[0]):
                import pdb; pdb.set_trace()
            active_additional_observed_action_probs[hist_idx, t_step] = active_infer_mdp.P(prev_state[0], active_infer_mdp.action_list[obs_act_idx],
                                                                 this_state[0])

    DataHelper.printStateHistories(active_additional_samples, env_mdp.observable_states)
    active_nominal_log_prob_data = (np.log(active_additional_observed_action_probs[:, 1:]).sum() +
                                     nominal_log_prob_data)

    if nominal_log_prob_data != 0.0:
        eps = 0.01 / (-nominal_log_prob_data)
    else:
        eps = 0.01 / (hist_idx + 1)
    active_infer_mdp.inferPolicy(method=inference_method, histories=run_histories, do_print=False,
                          theta_std_dev_0=active_infer_mdp.theta_std_dev, theta_0=active_infer_mdp.theta,
                          reference_policy_vec=true_env_policy_vec, monte_carlo_size=num_theta_samples,
                          print_iterations=False, eps=eps, velocity_memory=0.0, theta_std_dev_min=theta_std_dev_min,
                          theta_std_dev_max=np.inf, nominal_log_prob_data=nominal_log_prob_data,
                          moving_avg_min_slope=0.001, moving_average_buffer_length=60, do_plot=False,
                          precomputed_observed_action_indices=observed_action_indices,
                          additional_precomputed_observed_action_indices=active_additional_observed_action_indices,
                          additional_samples=active_additional_samples,
                          min_uncertainty=theta_std_dev_min)
    inferred_policy_L1_norm_error = MDP.getPolicyL1Norm(true_env_policy_vec, active_infer_mdp.getPolicyAsVec())
    print('ACTIVE with resamples: L1-norm as a fraction of max error: {}.'.format(inferred_policy_L1_norm_error/2/infer_mdp.num_states))
    active_policy_L1_norms.append(inferred_policy_L1_norm_error)
    active_policy_variance.append(np.sum(np.power(active_infer_mdp.theta_std_dev, 2)))

    return active_policy_L1_norms, active_policy_variance, passive_policy_L1_norms, passive_policy_variance


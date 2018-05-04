#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'


from NFA_DFA_Module.DFA import LTL_plus
from MDP_EM.MDP_EM.MDP import MDP
from MDP_EM.MDP_EM.inference_mdp import InferenceMDP
import MDP_EM.MDP_EM.plot_helper as PlotHelp
import MDP_EM.MDP_EM.data_helper as DataHelp
import experiment_configs as ExperimentConfigs

import time
import numpy as np
from copy import deepcopy
from pprint import pprint
from collections import OrderedDict
import itertools
import warnings
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

########################################################################################################################
# Numpy Print Options
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)

########################################################################################################################
# Grid, number of agents, obstacle, label, action, initial and goal state configuration

grid_dim = [8, 8] # [num-rows, num-cols]
grid_map = np.array(range(0,np.prod(grid_dim)), dtype=np.int8).reshape(grid_dim)

# Create a list of tuples where the tuples have length @c num_agents and represent the joint states of the agents.
num_agents = 1
states = [state for state in itertools.product(xrange(grid_map.size), repeat=num_agents)]
num_states = len(states)
state_indices = range(num_states)

# Atomic Proposition and labels configuration. Note that 'empty' is the empty string/label/dfa-action. The empty action
# for MDP incurrs a self loop. The alphabet dictionary is made of LTL_pluss atomic propositions.
green = LTL_plus('green')
red = LTL_plus('red')
empty = LTL_plus('E') # <-- 'E' is defined to be 'empty' in LTL_plus class.
alphabet_dict = {'empty': empty, 'green': green, 'red': red}
labels = {state: empty for state in states}
labels[(18,)] = green

# Starting and final states
initial_state = (53,)
goal_state = (18,) # Currently assumess only one goal.
labels[(21,)] = red
labels[(29,)] = red
labels[(41,)] = red
labels[(42,)] = red

# Numpy Data type to use for transition probability matrices (affects speed / precision)
prob_dtype = np.float64

# Numpy Data type to use for policy inference calculations (affects speed / precision). Less than 32 bits can result in
# 'NaN' values.
infer_dtype = np.float64

# Action options.
action_list = ['Empty', 'North', 'South', 'East', 'West']

# Load Transition Probability matricies
act_prob = ExperimentConfigs.getActionProbabilityDictionary(prob_dtype)

# Set `solve_with_uniform_distribution` to True to have the initial distribution for EM and the history/demonstration
# generation start with a uniform (MDP.S default) distribution across the values assigned to MDP.init_set. Set this to
# _False_ to have EM and the MDP always start from the `initial_state` below.
solve_with_uniform_distribution = True

#######################################################################################################################
# Entry point when called from Command line.
if __name__=='__main__':
    # Use single_agent_inference.py to infer a policy, and load that file from `pickled_inference` below. Then augment
    # the inferred policy with somme manually configured features and try to infer _that_ policy.
    pickled_mdp_file_to_load  = 'robot_mdps_180502_2038_HIST_5000eps10steps_180504_0902_Policy_180504_0932'

    # Demonstration history set of  episodes (aka trajectories) create/load options. If @c gather_new_data is false,
    # load the @c pickled_episodes_file. If @c gather_new_data is true, use @c num_episodes and @c steps_per_episode to
    # determine how large the demonstration set should be.
    gather_new_data = True
    #initial_traj_states = [0, 7, 56, 63]
    num_episodes = 500
    steps_per_episode = 10
    pickled_episodes_file_to_load = 'robot_mdps_180502_2038_HIST_5000eps10steps_180504_0902'

    # Perform/load policy inference options. If @c perform_new_inference is false, load the
    # @pickled_inference_mdps_file. The inference statistics files contain an array of L1-norm errors from the
    # demonstration policy.
    perform_new_inference = True
    pickled_inference_mdps_file_to_load  = \
        'robot_mdps_180424_2200_HIST_10eps10steps_180425_1723_Policy_180425_1723'
    # Select the inference method to use. Must match a method in PolicyInference: 'gradientAscent', 'historyMLE',
    # 'iterativeBayes', 'gradientAscentGaussianTheta'.
    inference_method = 'gradientAscentGaussianTheta'

    # Gradient Ascent kernel configurations
    kernel_centers =  frozenset(range(0, num_states, 5)) | frozenset([21,29,41,42])
    #kernel_centers = frozenset([0, 4, 12, 13, 14, 20, 24])
    #kernel_centers = frozenset(range(0, num_states, 1))
    #kernel_centers = frozenset((0, 4, 12, 20, 24))
    num_kernels= len(kernel_centers)
    kernel_sigmas = np.array([2.0]*num_kernels, dtype=infer_dtype)

    if inference_method is 'gradientAscentGaussianTheta':
        num_theta_samples = 1000
        monte_carlo_size = num_theta_samples
    else:
        monte_carlo_size = None

    # Plotting lags
    plot_all_grids = False
    plot_initial_mdp_grids = False
    plot_inferred_mdp_grids = True
    plot_demonstration = True
    plot_uncertainty = True
    plot_new_phi = False
    plot_new_kernel = False
    plot_loaded_phi = False
    plot_loaded_kernel = False
    plot_flags = [plot_all_grids, plot_initial_mdp_grids, plot_inferred_mdp_grids, plot_new_phi, plot_loaded_phi,
                  plot_new_kernel, plot_loaded_kernel, plot_demonstration, plot_uncertainty]
    if plot_new_kernel and plot_loaded_kernel:
        raise ValueError('Can not plot both new and loaded kernel in same call.')
    if plot_new_kernel:
        raise NotImplementedError('option: plot_new_kernel doesn\'t work yet. Sorry, but plot_new_phi works!')
    if plot_new_phi and plot_loaded_phi:
        raise ValueError('Can not plot both new and loaded phi in same call.')

    demo_mdp, pickled_mdp_file, = DataHelp.loadPickledPolicyInferenceMDP(pickled_mdp_file_to_load)
    policy_keys_to_print = demo_mdp.states
    reference_policy_vec = demo_mdp.getPolicyAsVec(policy_keys_to_print)
    #demo_mdp.computeErgodicCoefficient()

    # Ensure initial state for batch 0 will be uniformly, randomly selected.
    demo_mdp.init_set = policy_keys_to_print
    demo_mdp.setInitialProbDist(demo_mdp.init_set)

    if gather_new_data:
        # Use policy to simulate and record results.
        #
        # Start by simulating 10 steps each episode.
        hist_dtype = DataHelp.getSmallestNumpyUnsignedIntType(demo_mdp.num_states)
        run_histories = np.zeros([num_episodes, steps_per_episode], dtype=hist_dtype)
        #for episode, init_state in zip(xrange(num_episodes), initial_traj_states):
        #    # Create time-history for this episode.
        #    demo_mdp.current_state = policy_keys_to_print[init_state]
        #    run_histories[episode, 0] = init_state #As initial state
        for episode in range(num_episodes):
            # Create time-history for this episode.
            _, run_histories[episode, 0] = demo_mdp.resetState()
            for t_step in range(1, steps_per_episode):
                _, run_histories[episode, t_step] = demo_mdp.step()
        pickled_episodes_file = DataHelp.pickleEpisodes(variables_to_save=[run_histories], name_prefix=pickled_mdp_file,
                                                        num_episodes=num_episodes, steps_per_episode=steps_per_episode)
    else:
        # Load pickled episodes. Note that trailing comma on assignment automatically unpacks run_histories from a list.
        (run_histories, pickled_episodes_file) = DataHelp.loadPickledEpisodes(pickled_episodes_file_to_load)
        num_episodes = run_histories.shape[0]
        steps_per_episode = run_histories.shape[1]

    #DataHelp.printHistoryAnalysis(run_histories, states, labels, empty, goal_state)
    #DataHelp.printStateHistories(run_histories, mdp.observable_states)

    if plot_new_phi or  plot_new_kernel or perform_new_inference:
        tic = time.time()
        # Solve for approximated observed policy.
        # Use a new demo_mdp to model created/loaded one and a @ref GridGraph object to record, and seach for shortest paths
        # between two grid-cells.
        infer_mdp = InferenceMDP(init=initial_state, action_list=action_list, states=states,
                                 act_prob=deepcopy(act_prob), grid_map=grid_map, L=labels,
                                 gg_kernel_centers=kernel_centers, kernel_sigmas=kernel_sigmas, temp=0.3)
        print 'Built InferenceMDP with kernel set:'
        print(kernel_centers)
        if not perform_new_inference and (plot_new_phi or plot_new_kernel):
            # Deepcopy the infer_mdp to another variable because and old inference will be loaded into `infer_mdp`.
            new_infer_mdp = deepcopy(infer_mdp)
    if perform_new_inference:
        # Infer the policy from the recorded data.

        # Precompute observed actions for all episodes. Should do this in a "history" class.
        observation_dtype  = DataHelp.getSmallestNumpyUnsignedIntType(demo_mdp.num_actions)
        observed_action_indices = np.empty([num_episodes, steps_per_episode], dtype=observation_dtype)
        observed_action_probs = np.empty([num_episodes, steps_per_episode], dtype=infer_dtype)
        for episode in xrange(num_episodes):
            for t_step in xrange(1, steps_per_episode):
                this_state_idx = run_histories[episode, t_step-1]
                this_state = demo_mdp.observable_states[this_state_idx]
                next_state_idx = run_histories[episode, t_step]
                next_state = demo_mdp.observable_states[next_state_idx]
                this_state = (this_state,)
                next_state = (next_state,)
                # Packeage states in expected tuple-format
                obs_act_idx = infer_mdp.graph.getObservedAction(this_state, next_state)
                observed_action_indices[episode, t_step] = obs_act_idx
                observed_action_probs[episode, t_step] = infer_mdp.P(this_state[0], infer_mdp.action_list[obs_act_idx],
                                                                    next_state[0])

        # The nominal log probability of the trajectory data sets, if the observed action at each t-step was actually
        # the selected action.
        nominal_log_prob_data = np.log(observed_action_probs[:, 1:]).sum()

        print 'Nominal Log prob {}'.format(nominal_log_prob_data)
        if nominal_log_prob_data != 0.0:
            eps = 0.02 / (-nominal_log_prob_data)
        else:
            eps = 0.001
        theta_vec = infer_mdp.inferPolicy(method=inference_method, histories=run_histories, do_print=False,
                                          reference_policy_vec=reference_policy_vec, monte_carlo_size=monte_carlo_size,
                                          dtype=infer_dtype, print_iterations=True, eps=eps, velocity_memory=0.2,
                                          moving_avg_min_improvement=-np.inf,theta_std_dev_0=np.ones(num_kernels*5),
                                          theta_std_dev_max=np.inf, theta_std_dev_min=0.3,
                                          moving_average_buffer_length=400, moving_avg_min_slope=0.001,
                                          precomputed_observed_action_indices=observed_action_indices,
                                          nominal_log_prob_data=nominal_log_prob_data, use_precomputed_phi=True,
                                          max_uncertainty=0.0)
        toc = time.time() -tic
        print 'Total time to infer policy: {} sec, or {} min.'.format(toc, toc/60.0)
        pickled_inference_file = DataHelp.picklePolicyInferenceMDP(variables_to_save=[infer_mdp],
                                                                   name_prefix=pickled_episodes_file)
    else:
        (infer_mdp, pickled_inference_file) = \
            DataHelp.loadPickledPolicyInferenceMDP(pickled_inference_mdps_file_to_load)

    # Remember that variable @ref demo_mdp is used for demonstration.
    if len(policy_keys_to_print) == infer_mdp.num_states:
        infered_policy_L1_norm_error = MDP.getPolicyL1Norm(reference_policy_vec, infer_mdp.getPolicyAsVec())
        print('L1-norm between reference and inferred policy: {}.'.format(infered_policy_L1_norm_error))
        print('L1-norm as a fraction of max error: {}.'.format(infered_policy_L1_norm_error/2/num_states))
    else:
        warnings.warn('Demonstration MDP and inferred MDP do not have the same number of states. Perhaps one was '
                      'loaded from an old file? Not printing policy difference.')

    if any(plot_flags):
        maze, cmap = PlotHelp.PlotGrid.buildGridPlotArgs(grid_map, labels, alphabet_dict)

    plot_policies = []
    titles = []
    kernel_locations = []
    if plot_all_grids:
        plot_policies.append(demo_mdp.policy)
        plot_policies.append(infer_mdp.policy)
        titles = ['', '']
        kernel_locations = [None, infer_mdp.kernel_centers]
    elif plot_initial_mdp_grids:
        plot_policies.append(demo_mdp.policy)
        titles = ['']
        kernel_locations = [None]
    if plot_inferred_mdp_grids and not plot_all_grids:
        plot_policies.append(infer_mdp.policy)
        titles.append('')
        kernel_locations.append(infer_mdp.kernel_centers)

    if plot_all_grids or plot_initial_mdp_grids or plot_inferred_mdp_grids:
        center_offset = 0.5 # Shifts points into center of cell.
        base_policy_grid = PlotHelp.PlotPolicy(maze, cmap, center_offset)
        for policy, title, kernel_loc in zip(plot_policies, titles, kernel_locations):
            # Reorder policy dict for plotting.
            order_of_keys = [key for key in states]
            list_of_tuples = [(key, policy[key]) for key in order_of_keys]
            policy = OrderedDict(list_of_tuples)
            fig = base_policy_grid.configurePlot(title, policy, action_list, policy_keys_to_print=policy_keys_to_print,
                                                 decimals=3, kernel_locations=kernel_loc)

    if plot_demonstration:
        demo_grid = PlotHelp.PlotDemonstration(maze, cmap, y_center_offset=0.6, x_center_offset=0.3, fontsize=32)
        demo_grid.configurePlot('', run_histories)

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
            fig, ax = phi_grid.configurePlot(title, phi_idx, phi_at_state=phi_at_state, act=act,
                                             states=infer_mdp.states)

    if plot_uncertainty:
        # Only for GaussianTheta
        uncertainty_grid = PlotHelp.UncertaintyPlot(maze, cmap, grid_map)
        policy_uncertainty = infer_mdp.policy_uncertainty_as_vec.reshape([infer_mdp.num_states, infer_mdp.num_actions])
        for act_idx, act in enumerate(action_list):
            param_vector_indices = xrange(act_idx, len(infer_mdp.theta), len(action_list))
            uncertainty_vals = infer_mdp.theta_std_dev[param_vector_indices]
            title='Param Uncertainty'
            fig, ax = uncertainty_grid.configurePlot(title, infer_mdp.kernel_centers, uncertainty_vals, act_str=str(act))
            # Plot aggregate state-action uncertainty at states here
            fig, ax = uncertainty_grid.configurePlot('Policy Uncertainty', infer_mdp.grid_map.ravel(),
                                                     policy_uncertainty[:, act_idx], act_str=str(act))
        # Plot uncertainties at states (summed over actions)
        title = ''
        fig, ax = uncertainty_grid.configurePlot(title, infer_mdp.grid_map.ravel(), policy_uncertainty.sum(axis=1) ,
                                                 act_str='')

    if any(plot_flags):
        plt.show()
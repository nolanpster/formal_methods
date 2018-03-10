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
# Load Transition Probability matricies
act_prob = ExperimentConfigs.getActionProbabilityDictionary()
########################################################################################################################
# Grid, number of agents, obstacle, label, action, initial and goal state configuration

grid_dim = [4, 4] # [num-rows, num-cols]
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
labels[(6,)] = red
labels[(7,)] = red
labels[(8,)] = red
labels[(0,)] = green

# Starting and final states
initial_state = (15,)
goal_state = (0,) # Currently assumess only one goal.

# Action options.
action_list = ['Empty', 'North', 'South', 'East', 'West']

# Set `solve_with_uniform_distribution` to True to have the initial distribution for EM and the history/demonstration
# generation start with a uniform (MDP.S default) distribution across the values assigned to MDP.init_set. Set this to
# _False_ to have EM and the MDP always start from the `initial_state` below.
solve_with_uniform_distribution = False

#######################################################################################################################
# Entry point when called from Command line.
if __name__=='__main__':
    # MDP solution/load options. If @c make_new_mdp is false load the @c pickled_mdp_file.
    make_new_mdp = False
    pickled_mdp_file_to_load  = 'robot_mdps_180310_1124'
    write_mdp_policy_csv = False

    # Demonstration history set of  episodes (aka trajectories) create/load options. If @c gather_new_data is false,
    # load the @c pickled_episodes_file. If @c gather_new_data is true, use @c num_episodes and @c steps_per_episode to
    # determine how large the demonstration set should be.
    gather_new_data = False
    num_episodes = 250
    steps_per_episode = 15
    pickled_episodes_file_to_load = 'robot_mdps_180309_2256_HIST_250eps15steps_180309_2256'

    # Perform/load policy inference options. If @c perform_new_inference is false, load the
    # @pickled_inference_mdps_file. The inference statistics files contain an array of L1-norm errors from the
    # demonstration policy.
    perform_new_inference = True
    pickled_inference_mdps_file_to_load  = \
        'robot_mdps_180221_1237_HIST_250eps15steps_180221_1306_Policy_180221_1418'
    load_inference_statistics = (not perform_new_inference) & True
    pickled_inference_statistics_file_to_load  = \
        'robot_mdps_180221_1237_HIST_250eps15steps_180221_1306_Inference_Stats_180221_1431'
    # Select the inference method to use. Must match a method in PolicyInference: 'gradientAscent', 'historyMLE',
    # 'iterativeBayes', 'gradientAscentGaussianTheta'.
    inference_method = 'gradientAscentGaussianTheta'

    # Gradient Ascent kernel configurations
    use_fixed_kernel_set = True
    if use_fixed_kernel_set is True:
        kernel_centers = [frozenset([0, 2, 5, 7, 8, 10, 13, 15])]
        num_kernels_in_set = len(kernel_centers[0])
        kernel_sigmas = np.array([1.5]*num_kernels_in_set, dtype=np.float32)
    else:
        kernel_count_start = 10
        kernel_count_end = 9
        kernel_sigmas = np.array([1]*kernel_count_start, dtype=np.float32)
        kernel_count_increment_per_set = -1
        kernel_set_sample_count = 1
        batch_size_for_kernel_set = 1

    if inference_method is 'gradientAscentGaussianTheta':
        num_theta_samples = 8000
        monte_carlo_size = num_theta_samples
    else:
        monte_carlo_size = batch_size_for_kernel_set


    # Plotting lags
    plot_all_grids = False
    plot_initial_mdp_grids = False
    plot_inferred_mdp_grids = False
    plot_demonstration = True
    plot_uncertainty = False
    plot_new_phi = False
    plot_new_kernel = False
    plot_loaded_phi = False
    plot_loaded_kernel = False
    plot_inference_statistics = False
    plot_flags = [plot_all_grids, plot_initial_mdp_grids, plot_inferred_mdp_grids, plot_new_phi, plot_loaded_phi,
                  plot_new_kernel, plot_loaded_kernel, plot_inference_statistics, plot_demonstration, plot_uncertainty]
    if plot_new_kernel and plot_loaded_kernel:
        raise ValueError('Can not plot both new and loaded kernel in same call.')
    if plot_new_kernel:
        raise NotImplementedError('option: plot_new_kernel doesn\'t work yet. Sorry, but plot_new_phi works!')
    if plot_new_phi and plot_loaded_phi:
        raise ValueError('Can not plot both new and loaded phi in same call.')

    if use_fixed_kernel_set:
        num_kernel_sets = 1
        kernel_set_sample_count = 1
    else:
        # Configure kernel set iterations.
        num_kernels_in_set = np.arange(kernel_count_start, kernel_count_end, kernel_count_increment_per_set)
        num_kernel_sets = len(num_kernels_in_set)
        kernel_centers = {set_idx: frozenset(np.random.choice(len(states), num_kernels_in_set[set_idx] , replace=False))
                          for set_idx in range(num_kernel_sets)}
    kernel_set_L1_err = np.empty([num_kernel_sets, kernel_set_sample_count])
    kernel_set_infer_time = np.empty([num_kernel_sets, kernel_set_sample_count])

    if make_new_mdp:
        if solve_with_uniform_distribution:
            init_set = states
        else:
            # Leave MDP.init_set unset (=None) to solve the system from a single initial_state.
            init_set = None
        (EM_mdp, VI_mdp, policy_keys_to_print, policy_difference) = \
            ExperimentConfigs.makeGridMDPxDRA(states, initial_state, action_list, alphabet_dict, labels, grid_map,
                                              do_print=True, init_set=init_set)
        variables_to_save = [EM_mdp, VI_mdp, policy_keys_to_print, policy_difference]
        pickled_mdp_file = DataHelp.pickleMDP(variables_to_save, name_prefix="robot_mdps")
    else:
        (EM_mdp, VI_mdp, policy_keys_to_print, policy_difference, pickled_mdp_file) = \
            DataHelp.loadPickledMDP(pickled_mdp_file_to_load)
    if write_mdp_policy_csv:
        DataHelp.writePolicyToCSV(policy_difference, policy_keys_to_print,
                                  file_name=pickled_mdp_file+'_Policy_difference')
        DataHelp.writePolicyToCSV(EM_mdp.policy, policy_keys_to_print, file_name=pickled_mdp_file+'_EM_Policy')
        DataHelp.writePolicyToCSV(VI_mdp.policy, policy_keys_to_print, file_name=pickled_mdp_file+'_EM_Policy')

    # Choose which policy to use for demonstration.
    mdp = VI_mdp
    reference_policy_vec = mdp.getPolicyAsVec(policy_keys_to_print)

    if gather_new_data:
        # Use policy to simulate and record results.
        #
        # Current policy E{T|R} 6.7. Start by simulating 10 steps each episode.
        hist_dtype = DataHelp.getSmallestNumpyUnsignedIntType(mdp.num_states)
        run_histories = np.zeros([num_episodes, steps_per_episode], dtype=hist_dtype)
        for episode in range(num_episodes):
            # Create time-history for this episode.
            _, run_histories[episode, 0] = mdp.resetState()
            for t_step in range(1, steps_per_episode):
                _, run_histories[episode, t_step] = mdp.step()
        pickled_episodes_file = DataHelp.pickleEpisodes(variables_to_save=[run_histories], name_prefix=pickled_mdp_file,
                                                        num_episodes=num_episodes, steps_per_episode=steps_per_episode)
    else:
        # Load pickled episodes. Note that trailing comma on assignment automatically unpacks run_histories from a list.
        (run_histories, pickled_episodes_file) = DataHelp.loadPickledEpisodes(pickled_episodes_file_to_load)
        num_episodes = run_histories.shape[0]
        steps_per_episode = run_histories.shape[1]

    DataHelp.printHistoryAnalysis(run_histories, state_indices, labels, empty, goal_state)

    if plot_new_phi or  plot_new_kernel or perform_new_inference:
        tic = time.clock()
        # Solve for approximated observed policy.
        # Use a new mdp to model created/loaded one and a @ref GridGraph object to record, and seach for shortest paths
        # between two grid-cells.
        infer_mdp = InferenceMDP(init=initial_state, action_list=action_list, states=states,
                                 act_prob=deepcopy(act_prob), grid_map=grid_map, L=labels,
                                 gg_kernel_centers=kernel_centers[0], kernel_sigmas=kernel_sigmas)
        print 'Built InferenceMDP with kernel set:'
        print(kernel_centers[0])
        if not perform_new_inference and (plot_new_phi or plot_new_kernel):
            # Deepcopy the infer_mdp to another variable because and old inference will be loaded into `infer_mdp`.
            new_infer_mdp = deepcopy(infer_mdp)
    if perform_new_inference:
        # Infer the policy from the recorded data.

        # Precompute observed actions for all episodes. Should do this in a "history" class.
        observation_dtype  = DataHelp.getSmallestNumpyUnsignedIntType(mdp.num_actions)
        observed_action_indeces = np.empty([num_episodes, steps_per_episode], dtype=observation_dtype)
        for episode in xrange(num_episodes):
            for t_step in xrange(1, steps_per_episode):
                this_state = run_histories[episode, t_step-1]
                next_state = run_histories[episode, t_step]
                observed_action = infer_mdp.graph.getObservedAction(this_state, next_state)
                observed_action_indeces[episode, t_step] = action_list.index(observed_action)

        if inference_method in ['historyMLE', 'iterativeBayes', 'gradientAscentGaussianTheta']:
           theta_vec = infer_mdp.inferPolicy(method=inference_method, histories=run_histories,
                                             do_print=True, reference_policy_vec=reference_policy_vec,
                                             monte_carlo_size=monte_carlo_size)
        else:
            if batch_size_for_kernel_set > 1:
                print_inference_iterations = False
            else:
                print_inference_iterations = True

            # Peform the inference batch.
            for kernel_set_idx in xrange(num_kernel_sets):

                batch_L1_err = np.empty([kernel_set_sample_count, batch_size_for_kernel_set])
                batch_infer_time = np.empty([kernel_set_sample_count, batch_size_for_kernel_set])
                batch_kernel_sigmas = np.array([1]*num_kernels_in_set[kernel_set_idx], dtype=np.float32)
                for trial in xrange(kernel_set_sample_count):
                    trial_kernel_set = frozenset(np.random.choice(len(states), num_kernels_in_set[kernel_set_idx],
                                                 replace=False))
                    infer_mdp.buildKernels(gg_kernel_centers=trial_kernel_set, kernel_sigmas=batch_kernel_sigmas)
                    print('Inference set {} has {} kernels:{}'
                          .format((kernel_set_idx * kernel_set_sample_count) + trial,
                                  num_kernels_in_set[kernel_set_idx],
                                  infer_mdp.phi.kernel_centers))
                    batch_L1_err[trial], batch_infer_time[trial] = \
                        infer_mdp.inferPolicy(histories=run_histories,
                                              do_print=print_inference_iterations,
                                              use_precomputed_phi=True,
                                              dtype=np.float32,
                                              monte_carlo_size=monte_carlo_size,
                                              reference_policy_vec=reference_policy_vec,
                                              precomputed_observed_action_indeces=observed_action_indeces)
                kernel_set_L1_err[kernel_set_idx] = batch_L1_err.mean(axis=1)
                kernel_set_infer_time[kernel_set_idx] = batch_infer_time.mean(axis=1)
        toc = time.clock() -tic
        print 'Total time to infer policy{}: {} sec, or {} min.'.format(' set' if num_kernel_sets > 1 else '', toc,
                                                                        toc/60.0)
        pickled_inference_file = DataHelp.picklePolicyInferenceMDP(variables_to_save=[infer_mdp],
                                                                   name_prefix=pickled_episodes_file)
        if num_kernel_sets > 1:
            # Save data for inference sets
            pickled_inference_file = \
                DataHelp.pickleInferenceStatistics(variables_to_save=[kernel_set_L1_err, kernel_set_infer_time],
                                                   name_prefix=pickled_episodes_file)
    else:
        (infer_mdp, pickled_inference_file) = \
            DataHelp.loadPickledPolicyInferenceMDP(pickled_inference_mdps_file_to_load)

    if write_mdp_policy_csv:
        DataHelp.writePolicyToCSV(infer_mdp.policy, file_name=pickled_inference_file)

    if load_inference_statistics:
        (kernel_set_L1_err, kernel_set_infer_time, pickled_inference_statistics_file) = \
            DataHelp.loadPickledInferenceStatistics(pickled_inference_statistics_file_to_load)

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
                order_of_keys = [key for key in states]
                list_of_tuples = [(key, policy[key]) for key in order_of_keys]
            policy = OrderedDict(list_of_tuples)
            fig = base_policy_grid.configurePlot(title, policy, action_list, use_print_keys, policy_keys_to_print,
                                                 decimals=2, kernel_locations=kernel_loc)
            plt.savefig('{}_solved.tif'.format(title), dpi=400, transparent=False)

        print '\n\nHEY! You! With the face! (computers don\'t have faces) Mazimize figure window to correctly show ' \
                'arrow/dot size ratio!\n'

    if plot_demonstration:
        demo_grid = PlotHelp.PlotDemonstration(maze, cmap, center_offset=0.5)
        demo_grid.configurePlot('State Visit Count in Demonstration', run_histories)

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
            param_vector_indeces = xrange(act_idx, len(infer_mdp.theta), len(action_list))
            uncertainty_vals = infer_mdp.theta_std_dev[param_vector_indeces]
            title='Param Uncertainty'
            fig, ax = uncertainty_grid.configurePlot(title, infer_mdp.kernel_centers, uncertainty_vals, act_str=str(act))
            # Plot aggregate uncertainty at states here
            fig, ax = uncertainty_grid.configurePlot('Policy Uncertainty', infer_mdp.grid_map.ravel(),
                                                     policy_uncertainty[:, act_idx], act_str=str(act))

    if plot_inference_statistics:
        infer_mdp.inferPolicy(method='historyMLE', histories=run_histories, do_print=False)
        mle_L1_norm = MDP.getPolicyL1Norm(reference_policy_vec, infer_mdp.getPolicyAsVec())
        print 'MLE L1 error is  {0:.3f}.'.format(mle_L1_norm)
        title = '' # Set to empty to format title in external program.
        if num_kernels_in_set.size > 1:
            PlotHelp.plotPolicyErrorVsNumberOfKernels(kernel_set_L1_err, num_kernels_in_set, title, mle_L1_norm)
        else:
            warnings.warn('Only one kernel set, can\'t plot inferece statistics.')

    if any(plot_flags):
        plt.show()

#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'


from NFA_DFA_Module.DFA import LTL_plus
from MDP_EM.MDP_EM.MDP import MDP
from MDP_EM.MDP_EM.inference_mdp import InferenceMDP
import MDP_EM.MDP_EM.plot_helper as PlotHelper
import MDP_EM.MDP_EM.data_helper as DataHelper
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

grid_dim = [5, 5] # [num-rows, num-cols]
num_cells = np.prod(grid_dim)
cell_indeces = range(0, num_cells)
grid_map = np.array(cell_indeces, dtype=np.int8).reshape(grid_dim)

# Create a list of tuples where the tuples have length @c num_agents and represent the joint states of the agents.
num_agents = 2
robot_idx = 0
env_idx = 1
states = [state for state in itertools.product(xrange(grid_map.size), repeat=num_agents)]
num_states = len(states)
state_indices = range(num_states)

# Atomic Proposition and labels configuration. Note that 'empty' is the empty string/label/dfa-action. The empty action
# for MDP incurrs a self loop. The alphabet dictionary is made of LTL_pluss atomic propositions.
green = LTL_plus('green')
red = LTL_plus('red')
empty = LTL_plus('E') # <-- 'E' is defined to be 'empty' in LTL_plus class.
alphabet_dict = {'empty': empty, 'green': green, 'red': red}

# Set `solve_with_uniform_distribution` to True to have the initial distribution for EM and the history/demonstration
# generation start with a uniform (MDP.S default) distribution across the values assigned to MDP.init_set. Set this to
# _False_ to have EM and the MDP always start from the `initial_state` below.
solve_with_uniform_distribution = False
robot_initial_cell = 24
env_initial_cell = 4
initial_state = (robot_initial_cell, env_initial_cell)

# Currently assumes the robot only has one goal cell. Also, fixed obstacles only affect the robot.
robot_goal_cell = 6 # Currently assumess only one goal.
robot_goal_states = [(robot_goal_cell, cell) for cell in cell_indeces]

fixed_obstacle_cells = [13, 14, 20]

labels = {state: empty for state in states}
fixed_obs_labels = {state: empty for state in states}
for state in states:
    if state in robot_goal_states:
        labels[state] = green
    elif state[robot_idx] in fixed_obstacle_cells:
        labels[state] = red
        fixed_obs_labels[state] = red
    elif state[robot_idx] == state[env_idx]:
        labels[state] = red

# Numpy Data type to use for transition probability matrices (affects speed / precision)
prob_dtype = np.float32

# Numpy Data type to use for policy inference calculations (affects speed / precision). Less than 32 bits can result in
# 'NaN' values.
infer_dtype = np.float32

# Action options.
action_list = ['Empty', 'North', 'South', 'East', 'West']
num_grid_actions = len(action_list)
action_dict = {robot_idx: action_list, env_idx: action_list}
joint_action_list = [str(agent_idx) + '_'+ act for agent_idx in xrange(num_agents) for act in action_dict[agent_idx]]
robot_action_list = joint_action_list[(robot_idx * num_grid_actions) : (robot_idx * num_grid_actions) + num_grid_actions]
env_action_list = joint_action_list[(env_idx * num_grid_actions) : (env_idx * num_grid_actions) + num_grid_actions]

########################################################################################################################
# Flags for script control
########################################################################################################################
# MDP solution/load options. If @c make_new_mdp is false load the @c pickled_mdp_file.
make_new_mdp = False
pickled_mdp_file_to_load  = 'multi_agent_mdps_180314_1508'


# Demonstration history set of  episodes (aka trajectories) create/load options. If @c gather_new_data is false,
# load the @c pickled_episodes_file. If @c gather_new_data is true, use @c num_episodes and @c steps_per_episode to
# determine how large the demonstration set should be.
gather_new_data = False
print_history_analysis = False
num_episodes = 500
steps_per_episode = 10
pickled_episodes_file_to_load = 'multi_agent_mdps_180314_1508_HIST_500eps10steps_180314_1509'

# Perform/load policy inference options. If @c perform_new_inference is false, load the @pickled_inference_mdps_file.
perform_new_inference = False
pickled_inference_mdps_file_to_load  = 'robot_mdps_180311_1149_HIST_250eps15steps_180311_1149_Policy_180311_1149'
inference_method = 'gradientAscentGaussianTheta'
gg_kernel_centers = [0, 4, 12, 20, 24]

# Gaussian Theta params
num_theta_samples = 3000

# Plotting flags
plot_all_grids = False
plot_VI_mdp_grids = True
plot_EM_mdp_grids = False
plot_inferred_mdp_grids = False
plot_flags = [plot_all_grids, plot_VI_mdp_grids, plot_EM_mdp_grids, plot_inferred_mdp_grids]
########################################################################################################################
# Create / Load Multi Agent MDP
########################################################################################################################
if make_new_mdp:
    if solve_with_uniform_distribution:
        init_set = states
    else:
        # Leave MDP.init_set unset (=None) to solve the system from a single initial_state.
        init_set = None
    if 'gradientAscent' in inference_method:
        use_mobile_kernels = True
    else:
        use_mobile_kernels = False
    VI_mdp, policy_keys_to_print = ExperimentConfigs.makeMultiAgentGridMDPxDRA(states, initial_state, action_dict,
                                                                               alphabet_dict, labels, grid_map,
                                                                               do_print=False, init_set=init_set,
                                                                               prob_dtype=prob_dtype,
                                                                               fixed_obstacle_labels=fixed_obs_labels,
                                                                               use_mobile_kernels=use_mobile_kernels,
                                                                               gg_kernel_centers=gg_kernel_centers)
    variables_to_save = [VI_mdp, policy_keys_to_print]
    pickled_mdp_file = DataHelper.pickleMDP(variables_to_save, name_prefix="multi_agent_mdps")
else:
    (VI_mdp, policy_keys_to_print, pickled_mdp_file) = DataHelper.loadPickledMDP(pickled_mdp_file_to_load)

# Override recorded initial dist to be uniform
VI_mdp.init_set=policy_keys_to_print
VI_mdp.setInitialProbDist()
reference_policy_vec = VI_mdp.getPolicyAsVec(policy_keys_to_print)

########################################################################################################################
# Demonstrate Trajectories
########################################################################################################################
demo_mdp = VI_mdp
if gather_new_data:
    # Use policy to simulate and record results.
    #
    # Current policy E{T|R} 6.7. Start by simulating 10 steps each episode.
    hist_dtype = DataHelper.getSmallestNumpyUnsignedIntType(demo_mdp.num_observable_states)
    run_histories = np.zeros([num_episodes, steps_per_episode], dtype=hist_dtype)
    for episode in range(num_episodes):
        # Create time-history for this episode.
        _, run_histories[episode, 0] = demo_mdp.resetState()
        for t_step in range(1, steps_per_episode):
            _, run_histories[episode, t_step] = demo_mdp.step()
    pickled_episodes_file = DataHelper.pickleEpisodes(variables_to_save=[run_histories], name_prefix=pickled_mdp_file,
                                                      num_episodes=num_episodes, steps_per_episode=steps_per_episode)
else:
    # Load pickled episodes. Note that trailing comma on assignment automatically unpacks run_histories from a list.
    (run_histories, pickled_episodes_file) = DataHelper.loadPickledEpisodes(pickled_episodes_file_to_load)
    num_episodes = run_histories.shape[0]
    steps_per_episode = run_histories.shape[1]

if print_history_analysis:
    DataHelper.printHistoryAnalysis(run_histories, demo_mdp.observable_states, labels, empty, robot_goal_states)
    DataHelper.printStateHistories(run_histories, demo_mdp.observable_states)

########################################################################################################################
# Test Policy Inference
########################################################################################################################
infer_mdp = demo_mdp.infer_env_mdp
true_env_policy_vec = infer_mdp.getPolicyAsVec(policy_to_convert=VI_mdp.env_policy[env_idx])
if perform_new_inference:
    # Infer the policy from the recorded data.

    if 'GaussianTheta' in inference_method:
        monte_carlo_size = num_theta_samples
    else:
        monte_carlo_size = None

    # Precompute observed actions for all episodes. Should do this in a "history" class.
    observation_dtype  = DataHelper.getSmallestNumpyUnsignedIntType(demo_mdp.num_actions)
    observed_action_indeces = np.empty([num_episodes, steps_per_episode], dtype=observation_dtype)
    for episode in xrange(num_episodes):
        for t_step in xrange(1, steps_per_episode):
            this_state_idx = run_histories[episode, t_step-1]
            this_state = demo_mdp.observable_states[this_state_idx]
            next_state_idx = run_histories[episode, t_step]
            next_state = demo_mdp.observable_states[next_state_idx]
            observed_action_indeces[episode, t_step] = infer_mdp.graph.getObservedAction(this_state, next_state)

    theta_vec = infer_mdp.inferPolicy(method=inference_method, histories=run_histories, do_print=False,
                                     reference_policy_vec=true_env_policy_vec, use_precomputed_phi=True,
                                     monte_carlo_size=monte_carlo_size, print_iterations=True)

########################################################################################################################
# Print Results' analysis
########################################################################################################################
# Remember that variable @ref mdp is used for demonstration.

# Use the InferenceMDP to get the true policy vector since they have matching action keys.
if len(policy_keys_to_print) == infer_mdp.num_states:
    infered_policy_L1_norm_error = MDP.getPolicyL1Norm(true_env_policy_vec, infer_mdp.getPolicyAsVec())
    print('L1-norm between reference and inferred policy: {}.'.format(infered_policy_L1_norm_error))
    print('L1-norm as a fraction of max error: {}.'.format(2*infered_policy_L1_norm_error/len(true_env_policy_vec)))
else:
    warnings.warn('Demonstration MDP and inferred MDP do not have the same number of states. Perhaps one was '
                  'loaded from an old file? Not printing policy difference.')

########################################################################################################################
# Plot Results
########################################################################################################################
policy_keys_for_env_poses = {env_cell: [(robot_cell, env_cell) for robot_cell in demo_mdp.grid_cell_vec] for env_cell in
                             demo_mdp.grid_cell_vec}

if any(plot_flags):

    # Create plots for comparison. Note that the the `maze` array has one more row and column than the `grid` for
    # plotting purposes.
    maze = np.zeros(np.array(grid_dim)+1)
    for state, label in labels.iteritems():
        if label==red:
            grid_row, grid_col = np.where(grid_map==state[0])
            maze[grid_row, grid_col] = 2
        if label==green:
            grid_row, grid_col = np.where(grid_map==state[0])
            maze[grid_row, grid_col] = 1
    if red in labels.values():
        # Maximum value in maze corresponds to red.
        cmap = mcolors.ListedColormap(['white','green','red'])
    else:
        # Maximum value in maze corresponds to green.
        cmap = mcolors.ListedColormap(['white','green'])

    mdp_list, plot_policies, only_use_print_keys, titles, kernel_locations, action_lists = \
        PlotHelper.makePlotGroups(plot_all_grids, plot_VI_mdp_grids, plot_EM_mdp_grids, plot_inferred_mdp_grids,
                                  VI_mdp=VI_mdp, infer_mdp=infer_mdp, robot_action_list=robot_action_list,
                                  env_action_list=env_action_list)

    if plot_all_grids or plot_VI_mdp_grids or plot_EM_mdp_grids or plot_inferred_mdp_grids:
        center_offset = 0.5 # Shifts points into center of cell.
        base_policy_grid = PlotHelper.PlotPolicy(maze, cmap, center_offset)
        for mdp_to_plot, policy, use_print_keys, title, kernel_loc, act_list in zip(mdp_list, plot_policies,
                                                                                    only_use_print_keys, titles,
                                                                                    kernel_locations, action_lists):
            key_slicer = mdp_to_plot.cell_state_slicer
            # Reorder policy dict for plotting.
            if use_print_keys: # VI and EM policies have DRA states in policy keys.
                list_of_tuples = [(key[key_slicer], policy[key]) for key in policy_keys_to_print]
            else: # Learned policy only has state numbers.
                order_of_keys = [key for key in states]
                list_of_tuples = [(key[key_slicer], policy[key]) for key in order_of_keys]
            policy = OrderedDict(list_of_tuples)

            for env_pose in policy_keys_for_env_poses.keys():

                # Get policy at desired states to plot.
                list_of_tuples = [(key, policy[key]) for key in policy_keys_for_env_poses[env_pose]]
                policy_to_plot = OrderedDict(list_of_tuples)

                # Update the grid colors, assuming the environment is in a fixed locaiton.
                maze = np.zeros(np.array(grid_dim)+1)
                for state, label in labels.iteritems():
                    if state in policy_to_plot.keys():
                        if label==red:
                            if (state[robot_idx] in fixed_obstacle_cells) or (state[robot_idx] == state[env_idx]):
                                grid_row, grid_col = np.where(grid_map==state[robot_idx])
                                maze[grid_row, grid_col] = 2
                        if label==green:
                            grid_row, grid_col = np.where(grid_map==state[robot_idx])
                            maze[grid_row, grid_col] = 1
                base_policy_grid.updateCellColors(maze_cells=maze)

                fig = base_policy_grid.configurePlot(title, policy_to_plot, act_list, use_print_keys,
                                                     policy_keys_for_env_poses[env_pose], decimals=2,
                                                     kernel_locations=kernel_loc, stay_action=act_list[0])

        print '\n\nHEY! You! With the face! (computers don\'t have faces) Mazimize figure window to correctly show ' \
                'arrow/dot size ratio!\n'

plt.show()

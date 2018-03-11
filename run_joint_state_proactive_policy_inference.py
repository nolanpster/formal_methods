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
# Load Transition Probability matricies
robot_act_prob = ExperimentConfigs.getActionProbabilityDictionary()
env_act_prob = ExperimentConfigs.getActionProbabilityDictionary()

########################################################################################################################
# Grid, number of agents, obstacle, label, action, initial and goal state configuration

grid_dim = [3, 3] # [num-rows, num-cols]
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
orange = LTL_plus('orange')
red = LTL_plus('red')
empty = LTL_plus('E') # <-- 'E' is defined to be 'empty' in LTL_plus class.
alphabet_dict = {'empty': empty, 'green': green, 'red': red}

# Set `solve_with_uniform_distribution` to True to have the initial distribution for EM and the history/demonstration
# generation start with a uniform (MDP.S default) distribution across the values assigned to MDP.init_set. Set this to
# _False_ to have EM and the MDP always start from the `initial_state` below.
solve_with_uniform_distribution = False
robot_initial_cell = 8
env_initial_cell = 2
initial_state = (robot_initial_cell, env_initial_cell)

# Currently assumes the robot only has one goal cell. Also, fixed obstacles only affect the robot.
robot_goal_cell = 0 # Currently assumess only one goal.
robot_goal_states = [(robot_goal_cell, cell) for cell in cell_indeces]

fixed_obstacle_cells = [5]

labels = {state: empty for state in states}
for state in states:
    if state in robot_goal_states:
        labels[state] = green
    elif state[robot_idx] in fixed_obstacle_cells:
        labels[state] = red
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
make_new_mdp = True
pickled_mdp_file_to_load  = 'multi_agent_mdps_180311_2255'


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
    VI_mdp, policy_keys_to_print = ExperimentConfigs.makeMultiAgentGridMDPxDRA(states, initial_state, action_dict,
                                                                               alphabet_dict, labels, grid_map,
                                                                               do_print=True, init_set=init_set,
                                                                               prob_dtype=prob_dtype)
    variables_to_save = [VI_mdp, policy_keys_to_print]
    pickled_mdp_file = DataHelper.pickleMDP(variables_to_save, name_prefix="multi_agent_mdps")
else:
    (VI_mdp, policy_keys_to_print, pickled_mdp_file) = DataHelper.loadPickledMDP(pickled_mdp_file_to_load)


########################################################################################################################
# Plot Results
########################################################################################################################
policy_keys_for_env_poses = {env_cell: [(robot_cell, env_cell) for robot_cell in VI_mdp.grid_cell_vec] for env_cell in
                             VI_mdp.grid_cell_vec}

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

    mdp_list, plot_policies, only_use_print_keys, titles, kernel_locations = \
        PlotHelper.makePlotGroups(plot_all_grids, plot_VI_mdp_grids, plot_EM_mdp_grids, plot_inferred_mdp_grids, VI_mdp)

    if plot_all_grids or plot_VI_mdp_grids or plot_EM_mdp_grids or plot_inferred_mdp_grids:
        center_offset = 0.5 # Shifts points into center of cell.
        base_policy_grid = PlotHelper.PlotPolicy(maze, cmap, center_offset)
        for mdp_to_plot, policy, use_print_keys, title, kernel_loc in zip(mdp_list, plot_policies, only_use_print_keys,
                                                                          titles, kernel_locations):
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
                list_of_tuples = [(key, policy[(key,)]) for key in policy_keys_for_env_poses[env_pose]]
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

                fig = base_policy_grid.configurePlot(title, policy_to_plot, robot_action_list, use_print_keys,
                                                     policy_keys_for_env_poses[env_pose], decimals=2,
                                                     kernel_locations=kernel_loc, stay_action='0_Empty')

        print '\n\nHEY! You! With the face! (computers don\'t have faces) Mazimize figure window to correctly show ' \
                'arrow/dot size ratio!\n'

plt.show()

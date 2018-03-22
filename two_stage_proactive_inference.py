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

grid_dim = [3, 3] # [num-rows, num-cols]
num_cells = np.prod(grid_dim)
cell_indices = range(0, num_cells)
grid_map = np.array(cell_indices, dtype=np.int8).reshape(grid_dim)

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
robot_initial_cell = 8
env_initial_cell = 2
initial_state = (robot_initial_cell, env_initial_cell)

# Currently assumes the robot only has one goal cell. Also, fixed obstacles only affect the robot.
robot_goal_cell = 0 # Currently assumess only one goal.
robot_goal_states = [(robot_goal_cell, cell) for cell in cell_indices]

fixed_obstacle_cells = [5]

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
make_new_mdp = True
pickled_mdp_file_to_load  = 'multi_agent_mdps_180314_1428'


# Geodesic Gaussian Kernel centers
gg_kernel_centers = [0, 2, 4, 6, 8]

# Gaussian Theta params
num_theta_samples = 3000


########################################################################################################################
# Create / Load Multi Agent MDP
#
# Note: Intial Policy assumes environment has random walk across env_action_list.
########################################################################################################################
if make_new_mdp:
    if solve_with_uniform_distribution:
        init_set = states
    else:
        # Leave MDP.init_set unset (=None) to solve the system from a single initial_state.
        init_set = None
    VI_mdp, policy_keys_to_print = ExperimentConfigs.makeMultiAgentGridMDPxDRA(states, initial_state, action_dict,
                                                                               alphabet_dict, labels, grid_map,
                                                                               do_print=False, init_set=init_set,
                                                                               prob_dtype=prob_dtype,
                                                                               fixed_obstacle_labels=fixed_obs_labels,
                                                                               use_mobile_kernels=True,
                                                                               gg_kernel_centers=gg_kernel_centers)
    variables_to_save = [VI_mdp, policy_keys_to_print]
    pickled_mdp_file = DataHelper.pickleMDP(variables_to_save, name_prefix="multi_agent_mdps")
else:
    (VI_mdp, policy_keys_to_print, pickled_mdp_file) = DataHelper.loadPickledMDP(pickled_mdp_file_to_load)

########################################################################################################################
# Run Batch Inference
########################################################################################################################
ExperimentConfigs.rolloutInferSolve(VI_mdp, robot_idx, env_idx, num_batches=10, num_trajectories_per_batch=100,
        num_steps_per_traj=15, inference_method='gradientAscentGaussianTheta', infer_dtype=np.float64,
        num_theta_samples=2000, SGA_eps=0.00001, SGA_log_prob_thresh=np.log(0.8))

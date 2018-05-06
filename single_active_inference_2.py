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

grid_dim = [32, 32] # [num-rows, num-cols]
num_cells = np.prod(grid_dim)
cell_indices = range(0, num_cells)
grid_map = np.array(cell_indices).reshape(grid_dim)

# Create a list of tuples where the tuples have length @c num_agents and represent the joint states of the agents.
num_agents = 1
robot_idx = None
env_idx = 0
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
env_initial_cell = 20
initial_state = (env_initial_cell,)

fixed_obstacle_cells = [325,326,327,328]

labels = {state: empty for state in states}
env_labels = {state: empty for state in states}
fixed_obs_labels = {state: empty for state in states}
env_fixed_obs_labels = {state: empty for state in states}
for state in states:
    if state[env_idx] in fixed_obstacle_cells:
        labels[state] = red
        fixed_obs_labels[state] = red
    elif state[env_idx] in fixed_obstacle_cells:
        env_fixed_obs_labels[state] = red

# Numpy Data type to use for transition probability matrices (affects speed / precision)
prob_dtype = np.float64

# Numpy Data type to use for policy inference calculations (affects speed / precision). Less than 32 bits can result in
# 'NaN' values.
infer_dtype = np.float64

# Action options.
action_list = ['Empty', 'North', 'South', 'East', 'West']

# Load Transition Probability matricies
act_prob = ExperimentConfigs.getActionProbabilityDictionary(prob_dtype)

########################################################################################################################
# Flags for script control
########################################################################################################################
# MDP solution/load options. If @c make_new_mdp is false load the @c pickled_mdp_file.
make_new_mdp = False
pickled_mdp_file_to_load  = 'robot_mdps_180505_1841'
act_cost = 0.0


# Geodesic Gaussian Kernel centers
#gg_kernel_centers = range(0, num_cells, 1)
#gg_kernel_centers = [0, 4, 12, 20, 24]
#gg_kernel_centers = range(0, num_cells, 4) + [6, 18]
#gg_kernel_centers = frozenset(range(1, num_states, 2)) | frozenset([13,14, 20])
gg_kernel_centers = frozenset(range(0, num_states, 5)) | frozenset([21,29,41,42])
# Configure a grid of kernels with even spaceing
row_interval = 5
row_start = 7
kernel_rows = np.arange(row_start, grid_dim[0], row_interval)
num_kernel_rows = kernel_rows.size
kernel_rows = kernel_rows.reshape(num_kernel_rows, 1) # Reshape so we can broadcast to make @ref kernel_grid
kernel_rows *= grid_dim[1] # Cell number at start of each row

col_interval = 5
col_start = 7
kernel_cols = np.arange(col_start, grid_dim[1], col_interval)

kernel_grid = kernel_rows + kernel_cols

gg_kernel_centers = frozenset(kernel_grid.ravel()) | frozenset([325,326,327,328])

#gg_kernel_centers = frozenset([0, 4, 12, 13, 14, 20, 24])
num_kernels_in_set = len(gg_kernel_centers)
kernel_sigmas = np.array([5.5]*num_kernels_in_set, dtype=infer_dtype)

# Gaussian Theta params
use_active_inference = True
num_theta_samples = 1000
inference_temp = 0.4

# Batch configurations
num_batches = 10
traj_count_per_batch = 5
traj_length = 5
num_experiment_trials = 10
########################################################################################################################
# Create / Load Multi Agent MDP
#
# Note: Intial Policy assumes environment has random walk across env_action_list.
########################################################################################################################
if make_new_mdp:
    raise ValueError("Making new mdps in this script Not working, build it with the other single agent script.")
    if solve_with_uniform_distribution:
        init_set = states
    else:
        # Leave MDP.init_set unset (=None) to solve the system from a single initial_state.
        init_set = None
    (EM_mdp, VI_mdp, policy_keys_to_print, policy_difference) = \
        ExperimentConfigs.makeGridMDPxDRA(states, initial_state, action_list, alphabet_dict, labels, grid_map,
                                          do_print=True, init_set=init_set, prob_dtype=prob_dtype)
    demo_mdp = EM_mdp
    variables_to_save = [demo_mdp, policy_keys_to_print]
    pickled_mdp_file = DataHelper.pickleMDP(variables_to_save, name_prefix="multi_agent_mdps")
else:
    (EM_mdp, VI_mdp, policy_keys_to_print, policy_difference, pickled_mdp_file) = \
        DataHelper.loadPickledMDP(pickled_mdp_file_to_load)
    demo_mdp = EM_mdp

# Also update the kernels if they've changed.
infer_mdp = InferenceMDP(init=initial_state, action_list=action_list, states=states,
                         act_prob=deepcopy(act_prob), grid_map=grid_map, L=labels,
                         gg_kernel_centers=gg_kernel_centers, kernel_sigmas=kernel_sigmas, temp=inference_temp)

########################################################################################################################
# Run Batch Inference
########################################################################################################################
policy_L1_norm_sets = []
parameter_variances = []

# The initial guess is a uniform policy.
initial_guess_of_env_policy = deepcopy(infer_mdp.policy)

for trial in range(num_experiment_trials):
    print '\n'
    print 'Starting Trial {} of {}.'.format(trial+1, num_experiment_trials)
    print '\n'

    # Everytime we start a new batch ensure that the Robot starts with the same initial policy, and that it starts with 
    # the same initial guess of environment policy.
    infer_mdp.policy = deepcopy(initial_guess_of_env_policy)

    # Hand over data to experiment-runner.
    policy_L1_norms, parameter_variance = \
        ExperimentConfigs.rolloutInferSingleAgent(demo_mdp, infer_mdp, num_batches=num_batches,
                                                  num_trajectories_per_batch=traj_count_per_batch,
                                                  num_steps_per_traj=traj_length,
                                                  inference_method='gradientAscentGaussianTheta',
                                                  infer_dtype=infer_dtype, num_theta_samples=num_theta_samples,
                                                  robot_goal_states=None, use_active_inference=use_active_inference)

    policy_L1_norm_sets.append(policy_L1_norms)
    parameter_variances.append(parameter_variance)

# Stack the recorded statistics list in [num_batches, num_trials] format for easy plotting.
policy_L1_norms_mat = np.stack(policy_L1_norm_sets, axis=1)
policy_L1_norms_mat /= 2
policy_L1_norms_mat /= infer_mdp.num_states

parameter_variance_mat = np.stack(parameter_variances, axis=1)

# Save data for plotting later
if use_active_inference:
    generated_data = {'active_inference_L1_norms': policy_L1_norms_mat,
                      'active_inference_parameter_variance': parameter_variance_mat}
else:
    generated_data = {'passive_inference_L1_norms': policy_L1_norms_mat,
                      'passive_inference_parameter_variance': parameter_variance_mat}


inference_type_str = 'active' if use_active_inference else 'passive'
file_name_prefix = 'single_agent_{}_stats_{}_trials{}_batches_{}_trajs_{}_stepsPerTraj'.format(inference_type_str,
    num_experiment_trials, num_batches, traj_count_per_batch, traj_length)

DataHelper.pickleInferenceStatistics(generated_data, file_name_prefix)

PlotHelper.plotValueVsBatch(policy_L1_norms_mat,
    '{} Fractional L1 Norm Inference Error'.format('Active' if use_active_inference else 'Passive'), ylabel=None,
    xlabel='Batch', also_plot_stats=True, save_figures=False)

PlotHelper.plotValueVsBatch(parameter_variance_mat,
    '{} Total Parameter Variance'.format('Active' if use_active_inference else 'Passive'), ylabel=None,
    xlabel='Batch', also_plot_stats=True, save_figures=False)

plt.show()

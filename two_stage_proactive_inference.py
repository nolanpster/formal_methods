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
robot_initial_cell = 24
env_initial_cell = 20
initial_state = (robot_initial_cell, env_initial_cell)

# Currently assumes the robot only has one goal cell. Also, fixed obstacles only affect the robot.
robot_goal_cell = 0 # Currently assumess only one goal.
robot_goal_states = [(robot_goal_cell, cell) for cell in cell_indices]

fixed_obstacle_cells = []

labels = {state: empty for state in states}
env_labels = {state: empty for state in states}
fixed_obs_labels = {state: empty for state in states}
env_fixed_obs_labels = {state: empty for state in states}
for state in states:
    if state in robot_goal_states:
        labels[state] = green
    elif state[robot_idx] in fixed_obstacle_cells:
        labels[state] = red
        fixed_obs_labels[state] = red
    elif state[env_idx] in fixed_obstacle_cells:
        env_fixed_obs_labels[state] = red
    elif state[robot_idx] == state[env_idx]:
        labels[state] = red
        env_labels[state] = red

# Numpy Data type to use for transition probability matrices (affects speed / precision)
prob_dtype = np.float64

# Numpy Data type to use for policy inference calculations (affects speed / precision). Less than 32 bits can result in
# 'NaN' values.
infer_dtype = np.float64

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
pickled_mdp_file_to_load  = 'multi_agent_mdps_180423_1611'
act_cost =  0.0

true_optimal_policies_to_load = 'true_optimal_policies_em_15H_100N_Inference_Stats_180423_2008'

# Geodesic Gaussian Kernel centers
gg_kernel_centers = range(0, num_cells, 1)
gg_kernel_centers = [0, 4, 12, 20, 24, 24]  # Last kernel is the 'mobile' kernel
gg_kernel_centers = range(0, num_cells, 4) + [6, 18] + [24]
#gg_kernel_centers = range(0, num_cells, 1) + [24]
num_kernels_in_set = len(gg_kernel_centers)
kernel_sigmas = np.array([2.0]*num_kernels_in_set, dtype=infer_dtype)
ggk_mobile_indices = [num_kernels_in_set-1]

# Gaussian Theta params
use_active_inference = True
num_theta_samples = 1000
inference_temp = 0.5

# Batch configurations
num_batches = 100
traj_count_per_batch = 10
traj_length = 10
num_experiment_trials = 10
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
    demo_mdp, policy_keys_to_print = ExperimentConfigs.makeMultiAgentGridMDPxDRA(states, initial_state, action_dict,
                                                                                 alphabet_dict, labels, grid_map,
                                                                                 do_print=False, init_set=init_set,
                                                                                 prob_dtype=prob_dtype,
                                                                                 fixed_obstacle_labels=fixed_obs_labels,
                                                                                 use_mobile_kernels=True,
                                                                                 gg_kernel_centers=gg_kernel_centers,
                                                                                 use_em=True, act_cost=act_cost,
                                                                                 env_labels=env_labels)
    variables_to_save = [demo_mdp, policy_keys_to_print]
    pickled_mdp_file = DataHelper.pickleMDP(variables_to_save, name_prefix="multi_agent_mdps")
else:
    (demo_mdp, policy_keys_to_print, pickled_mdp_file) = DataHelper.loadPickledMDP(pickled_mdp_file_to_load)

# Also update the kernels if they've changed.
demo_mdp.infer_env_mdp.buildKernels(gg_kernel_centers=gg_kernel_centers, kernel_sigmas=kernel_sigmas,
                                    ggk_mobile_indices=ggk_mobile_indices, state_idx_of_mobile_kernel=0)
# Set the temperature to the desired value since it might be different than the one that the InferenceMDP was built
# with.
demo_mdp.infer_env_mdp.temp = inference_temp

# Override recorded initial dist to be uniform. Note that policy_keys_to_print are the reachable initial states, and we
# want to set the initial state-set to only include the states where the robot is at `robot_initial_cell`.

#demo_mdp.init_set = ((24, 20),)
#demo_mdp.setInitialProbDist(demo_mdp.init_set)

# The original environment policy in the MDP is a random walk. So we load a file containing a more interesting
# environent policy (generated in a single agent environment) then copy it into the joint state-space. Additionally, we
# add an additional feature vector to the environment's Q-function that represents how much the environment is repulsed
# by the robot. (Repulsive factors are buried in the method below). The method below updates the demo_mdp.env_policy
# dictionary.
ExperimentConfigs.convertSingleAgentEnvPolicyToMultiAgent(demo_mdp, labels, state_env_idx=env_idx,
                                                          new_kernel_weight=1.0, new_phi_sigma=1.0, plot_policies=False,
                                                          alphabet_dict=alphabet_dict,
                                                          fixed_obstacle_labels=fixed_obs_labels)


# Copy initial robot policy and initial guess for env policy. We'll reset the demo-mdp's variables to these each time
# to make sure that each trial has the same initial conditions.
robot_policy_0 = deepcopy(demo_mdp.policy)
initial_guess_of_env_policy = deepcopy(demo_mdp.infer_env_mdp.policy)

# Load optimal robot policy (given true env policy -- make sure this file matches with the one loaded by
# 'convertSingleAgentEnvPolicyToMultiAgent').
true_optimal_VI_policy, true_optimal_EM_policy, _, = \
    DataHelper.loadPickledInferenceStatistics(true_optimal_policies_to_load)

########################################################################################################################
# Run Batch Inference
########################################################################################################################
policy_L1_norm_sets = []
reward_count_sets = []
parameter_variances = []
bonus_reward_mags = []
robot_policy_L1_norm_sets = []

for trial in range(num_experiment_trials):
    print '\n'
    print 'Starting Trial {} of {}.'.format(trial + 1, num_experiment_trials)
    print '\n'

    # Everytime we start a new batch ensure that the Robot starts with the same initial policy, and that it starts with 
    # the same initial guess of environment policy.
    demo_mdp.policy = deepcopy(robot_policy_0)
    demo_mdp.infer_env_mdp.policy = deepcopy(initial_guess_of_env_policy)

    # Hand over data to experiment-runner.
    policy_L1_norms, reward_counts, parameter_variance, bonus_reward_mag, robot_policy_L1_norms = \
        ExperimentConfigs.rolloutInferSolve(demo_mdp, robot_idx, env_idx, num_batches=num_batches,
                                            num_trajectories_per_batch=traj_count_per_batch,
                                            num_steps_per_traj=traj_length,
                                            inference_method='gradientAscentGaussianTheta', infer_dtype=infer_dtype,
                                            num_theta_samples=num_theta_samples, robot_goal_states=robot_goal_states,
                                            act_cost=act_cost, use_active_inference=use_active_inference,
                                            true_optimal_VI_policy=true_optimal_VI_policy)
    policy_L1_norm_sets.append(policy_L1_norms)
    reward_count_sets.append(reward_counts)
    parameter_variances.append(parameter_variance)
    bonus_reward_mags.append(bonus_reward_mag)
    robot_policy_L1_norm_sets.append(robot_policy_L1_norms)

#Probably save the arrays? Figure out how to get passive and active on same plot.

# Stack the recorded statistics list in [num_batches, num_trials] format for easy plotting.
policy_L1_norms_mat = np.stack(policy_L1_norm_sets, axis=1)
policy_L1_norms_mat /= 2
policy_L1_norms_mat /= demo_mdp.num_states
reward_count_mat = np.stack(reward_count_sets, axis=1)
parameter_variance_mat = np.stack(parameter_variances, axis=1)
bonus_reward_mat = np.stack(bonus_reward_mags, axis=1)
robot_policy_L1_norm_mat = np.stack(robot_policy_L1_norm_sets, axis=1)

# Save data for plotting later
if use_active_inference:
    generated_data = {'active_inference_L1_norms': policy_L1_norms_mat,
                      'active_inference_count_of_trajs_reacing_goal': reward_count_mat,
                      'active_inference_parameter_variance': parameter_variance_mat,
                      'active_inference_bonus_reward': bonus_reward_mat,
                      'active_inference_robot_policy_err': robot_policy_L1_norm_mat}
else:
    generated_data = {'passive_inference_L1_norms': policy_L1_norms_mat,
                      'passive_inference_count_of_trajs_reacing_goal': reward_count_mat,
                      'passive_inference_parameter_variance': parameter_variance_mat,
                      'passive_inference_bonus_reward': bonus_reward_mat,
                      'passive_inference_robot_policy_err': robot_policy_L1_norm_mat}


inference_type_str = 'active' if use_active_inference else 'passive'
file_name_prefix = 'two_stage_{}_stats_{}_trials{}_batches_{}_trajs_{}_stepsPerTraj'.format(inference_type_str,
    num_experiment_trials, num_batches, traj_count_per_batch, traj_length)

DataHelper.pickleInferenceStatistics(generated_data, file_name_prefix)

PlotHelper.plotValueVsBatch(policy_L1_norms_mat,
    '{} Fractional L1 Norm Inference Error'.format('Active' if use_active_inference else 'Passive'), ylabel=None,
    xlabel='Batch', also_plot_stats=True, save_figures=False)

PlotHelper.plotValueVsBatch(reward_count_mat,
    '{} Total Trajectories Earning Rewards'.format('Active' if use_active_inference else 'Passive'), ylabel=None,
    xlabel='Batch', also_plot_stats=True, save_figures=False)

PlotHelper.plotValueVsBatch(parameter_variance_mat,
    '{} Total Parameter Variance'.format('Active' if use_active_inference else 'Passive'), ylabel=None,
    xlabel='Batch', also_plot_stats=True, save_figures=False)

PlotHelper.plotValueVsBatch(robot_policy_L1_norm_mat,
    '{} Robot Policy L_infty - Norm'.format('Active' if use_active_inference else 'Passive'), ylabel=None,
    xlabel='Batch', also_plot_stats=True, save_figures=False)

if use_active_inference:
    PlotHelper.plotValueVsBatch(bonus_reward_mat, 'Available Bonus Reward', ylabel=None, xlabel='Batch',
                                also_plot_stats=True, save_figures=False)

plt.show()

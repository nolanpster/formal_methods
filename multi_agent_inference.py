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
solve_EM = False
resolve_VI = False
solve_after_inference = False
pickled_mdp_file_to_load  = 'multi_agent_mdps_180408_1502'


# Demonstration history set of  episodes (aka trajectories) create/load options. If @c gather_new_data is false,
# load the @c pickled_episodes_file. If @c gather_new_data is true, use @c num_episodes and @c steps_per_episode to
# determine how large the demonstration set should be.
gather_new_data = False
print_history_analysis = False
num_episodes = 500
steps_per_episode = 10
pickled_episodes_file_to_load = 'multi_agent_mdps_180408_1502_HIST_500eps10steps_180408_1550'

# Perform/load policy inference options. If @c perform_new_inference is false, load the @pickled_inference_mdps_file.
perform_new_inference = True
pickled_two_stage_mdps_file_to_load  = 'two_stage_multi_agent_mdps_180408_1424'
inference_method = 'gradientAscentGaussianTheta'
gg_kernel_centers = [0, 4, 12, 20, 24, 24]  # Last kernel is the 'mobile' kernel
gg_kernel_centers = range(0, num_cells, 1) + [24]
num_kernels_in_set = len(gg_kernel_centers)
kernel_sigmas = np.array([1.0]*num_kernels_in_set, dtype=infer_dtype)
ggk_mobile_indices = [num_kernels_in_set-1]

# Gaussian Theta params
inference_temp = 0.8
num_theta_samples = 3000

# Plotting flags
plot_all_grids = False
plot_VI_mdp_grids = True
plot_EM_mdp_grids = False
plot_inferred_mdp_grids = False
plot_uncertainty = False
plot_flags = [plot_all_grids, plot_VI_mdp_grids, plot_EM_mdp_grids, plot_inferred_mdp_grids, plot_uncertainty]
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
                                                                               gg_kernel_centers=gg_kernel_centers,
                                                                               env_labels=env_labels, act_cost=act_cost)
    variables_to_save = [VI_mdp, policy_keys_to_print]
    pickled_mdp_file = DataHelper.pickleMDP(variables_to_save, name_prefix="multi_agent_mdps")
else:
    (VI_mdp, policy_keys_to_print, pickled_mdp_file) = DataHelper.loadPickledMDP(pickled_mdp_file_to_load)

# Override recorded initial dist to be uniform. Note that policy_keys_to_print are the reachable initial states, and we
# want to set the initial state-set to only include the states where the robot is at `robot_initial_cell`.
VI_mdp.init_set = VI_mdp.states
VI_mdp.setInitialProbDist(VI_mdp.init_set)

# The original environment policy in the MDP is a random walk. So we load a file containing a more interesting
# environent policy (generated in a single agent environment) then copy it into the joint state-space. Additionally, we
# add an additional feature vector to the environment's Q-function that represents how much the environment is repulsed
# by the robot. (Repulsive factors are buried in the method below). The method below updates the VI_mdp.env_policy
# dictionary.
ExperimentConfigs.convertSingleAgentEnvPolicyToMultiAgent(VI_mdp, labels, state_env_idx=env_idx,
                                                          new_kernel_weight=1.0, new_phi_sigma=1.0, plot_policies=False,
                                                          alphabet_dict=alphabet_dict,
                                                          fixed_obstacle_labels=fixed_obs_labels)

if resolve_VI:
    tic = time.time()
    VI_mdp.solve(print_iterations=True)
    print "VI: {}sec".format(time.time() - tic)
VI_policy = VI_mdp.getPolicyAsVec()
EM_mdp = deepcopy(VI_mdp)
EM_mdp.makeUniformPolicy()
if solve_EM:
    em_stats = EM_mdp.solve(method='expectationMaximization', do_print=False, horizon_length=20, num_iters=10)
    EM_error = EM_mdp.getPolicyL1Norm(VI_policy, EM_mdp.getPolicyAsVec())
    print 'EM L1 error: {}'.format(EM_error)
    print em_stats
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
    executed_robot_actions = np.zeros([num_episodes, steps_per_episode], dtype=hist_dtype)
    for episode in range(num_episodes):
        # Create time-history for this episode.
        _, run_histories[episode, 0] = demo_mdp.resetState()
        for t_step in range(1, steps_per_episode):
            _, run_histories[episode, t_step], executed_robot_action = demo_mdp.step()
            executed_robot_actions[episode, t_step] = robot_action_list.index(executed_robot_action)
    pickled_episodes_file = DataHelper.pickleEpisodes(variables_to_save=[run_histories, executed_robot_actions],
                                                      name_prefix=pickled_mdp_file, num_episodes=num_episodes,
                                                      steps_per_episode=steps_per_episode)
else:
    # Load pickled episodes. Note that trailing comma on assignment automatically unpacks run_histories from a list.
    (run_histories, executed_robot_actions, pickled_episodes_file) = \
        DataHelper.loadPickledEpisodes(pickled_episodes_file_to_load)
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
    observed_action_indices = np.empty([num_episodes, steps_per_episode], dtype=observation_dtype)
    observed_action_probs = np.empty([num_episodes, steps_per_episode], dtype=infer_dtype)
    for episode in xrange(num_episodes):
        for t_step in xrange(1, steps_per_episode):
            this_state_idx = run_histories[episode, t_step-1]
            this_state = demo_mdp.observable_states[this_state_idx]
            next_state_idx = run_histories[episode, t_step]
            next_state = demo_mdp.observable_states[next_state_idx]
            observed_action_indices[episode, t_step] = infer_mdp.graph.getObservedAction(this_state, next_state)
            robot_act = robot_action_list[executed_robot_actions[episode, t_step]]
            env_act = env_action_list[observed_action_indices[episode, t_step]]
            # This is a bit of a hack since `type(demo_mdp.mdp)` is a MultiAgentMDP and that has an overloaded self.P
            # function to return the probability of the joint state transition given two actions.
            observed_action_probs[episode, t_step] = demo_mdp.mdp.P(this_state, robot_act, env_act, next_state)

    # The nominal log probability of the trajectory data sets, if the observed action at each t-step was actually
    # the selected action.
    nominal_log_prob_data = np.log(observed_action_probs[:, 1:]).sum()

    # Also update the kernels if they've changed.
    infer_mdp.buildKernels(gg_kernel_centers=gg_kernel_centers, kernel_sigmas=kernel_sigmas,
                           ggk_mobile_indices=ggk_mobile_indices, state_idx_of_mobile_kernel=0)

    # Set the temperature to the desired value since it might be different than the one that the InferenceMDP was built
    # with.
    infer_mdp.temp = inference_temp

    theta_vec = infer_mdp.inferPolicy(method=inference_method, histories=run_histories, do_print=False,
                                      reference_policy_vec=true_env_policy_vec, use_precomputed_phi=True,
                                      monte_carlo_size=monte_carlo_size, print_iterations=True, eps=0.0005,
                                      velocity_memory=0.2, theta_std_dev_min=0.5, theta_std_dev_max=1.2,
                                      nominal_log_prob_data=nominal_log_prob_data, moving_avg_min_slope=0.001,
                                      moving_average_buffer_length=60)
    pickled_mdp_file = DataHelper.pickleMDP([demo_mdp, policy_keys_to_print], name_prefix="two_stage_multi_agent_mdps")
else:
    (demo_mdp, policy_keys_to_print, pickled_episodes_file) = \
        DataHelper.loadPickledMDP(pickled_two_stage_mdps_file_to_load)
    infer_mdp = demo_mdp.infer_env_mdp

if solve_after_inference:
    # This will update the demo-mdp's policy and therefore its plots below.
    bonus_reward_dict = ExperimentConfigs.makeBonusReward(infer_mdp.policy_uncertainty)
    winning_reward = {act: 0.0 for act in demo_mdp.action_list}
    winning_reward['0_Empty'] = 1.0
    demo_mdp.configureReward(winning_reward, bonus_reward_at_state=bonus_reward_dict)

    demo_mdp.solve(do_print=False, method='valueIteration', print_iterations=True,
                   policy_keys_to_print=policy_keys_to_print, horizon_length=20, num_iters=40)
    variables_to_save = [demo_mdp, policy_keys_to_print]
    pickled_mdp_file = DataHelper.pickleMDP(variables_to_save, name_prefix="multi_agent_mdps_bonus_reward")
else:
    (demo_mdp, policy_keys_to_print, pickled_mdp_file) = \
        DataHelper.loadPickledMDP('multi_agent_mdps_bonus_reward_180330_1347')



policy_change = demo_mdp.getPolicyL1Norm(VI_policy, demo_mdp.getPolicyAsVec())
print 'Policy Change from Bonus Reward: {}'.format(policy_change)
#import pdb; pdb.set_trace()
#VI_mdp = demo_mdp

########################################################################################################################
# Print Results' analysis
########################################################################################################################
# Remember that variable @ref mdp is used for demonstration.

inferred_policy_vec = infer_mdp.getPolicyAsVec()
# Use the InferenceMDP to get the true policy vector since they have matching action keys.
if len(policy_keys_to_print) == infer_mdp.num_states:
    infered_policy_L1_norm_error = MDP.getPolicyL1Norm(true_env_policy_vec, inferred_policy_vec)
    print('L1-norm between reference and inferred policy: {}.'.format(infered_policy_L1_norm_error))
    print('L1-norm as a fraction of max error: {}.'.format(infered_policy_L1_norm_error/2/num_states))
else:
    warnings.warn('Demonstration MDP and inferred MDP do not have the same number of states. Perhaps one was '
                  'loaded from an old file? Not printing policy difference.')

########################################################################################################################
# Plot Results
########################################################################################################################
VI_policy_key_groups = {env_cell: [((robot_cell, env_cell),) for robot_cell in demo_mdp.grid_cell_vec] for env_cell in
                             demo_mdp.grid_cell_vec}
infer_policy_key_groups = {robot_cell: [(robot_cell, env_cell) for env_cell in demo_mdp.grid_cell_vec] for robot_cell in
                            demo_mdp.grid_cell_vec}

if any(plot_flags):

    maze, cmap = PlotHelper.PlotGrid.buildGridPlotArgs(grid_map, labels, alphabet_dict, num_agents=2,
            fixed_obstacle_labels=fixed_obs_labels, agent_idx=0, goal_states=robot_goal_states)

    mdp_list, plot_policies, only_use_print_keys, titles, kernel_locations, action_lists, plot_key_groups = \
        PlotHelper.makePlotGroups(plot_all_grids, plot_VI_mdp_grids, plot_EM_mdp_grids, plot_inferred_mdp_grids,
                                  VI_mdp=VI_mdp, EM_mdp=EM_mdp, infer_mdp=infer_mdp,
                                  robot_action_list=robot_action_list, env_action_list=env_action_list,
                                  VI_plot_keys=VI_policy_key_groups, EM_plot_keys=VI_policy_key_groups,
                                  infer_plot_keys=infer_policy_key_groups, include_kernels=True)

    if plot_all_grids or plot_VI_mdp_grids or plot_EM_mdp_grids or plot_inferred_mdp_grids:
        center_offset = 0.5 # Shifts points into center of cell.
        base_policy_grid = PlotHelper.PlotPolicy(maze, cmap, center_offset)
        for mdp_to_plot, policy, use_print_keys, title, kernel_loc, act_list, plot_key_dict in \
                zip(mdp_list, plot_policies, only_use_print_keys, titles, kernel_locations, action_lists,
                    plot_key_groups):
            # Plot one policy plot for each environment location for the robot's policy, or plot one policy plot for
            # each robot location for the inference policy.

            key_slicer = mdp_to_plot.cell_state_slicer
            # Reorder policy dict for plotting.
            if use_print_keys: # VI and EM policies have DRA states in policy keys.
                list_of_tuples = [(key[key_slicer], policy[key]) for key in policy_keys_to_print]
            else: # Learned policy only has state numbers.
                order_of_keys = [key for key in states]
                list_of_tuples = [(key[key_slicer], policy[key]) for key in order_of_keys]
            policy = OrderedDict(list_of_tuples)

            # ID The fixed index for each plot group.
            if type(mdp_to_plot) is InferenceMDP:
                fixed_idx = 0 # Robot pose is fixed in plots below.
            else:
                fixed_idx = 1 # Env pose is fixed in plots below.

            for pose in plot_key_dict.keys():

                # Get policy at desired states to plot.
                list_of_tuples = [(key, policy[key]) for key in plot_key_dict[pose]]
                policy_to_plot = OrderedDict(list_of_tuples)

                # Update the grid colors, assuming the environment is in a fixed locaiton.
                this_maze = deepcopy(maze)
                for state, label in labels.iteritems():
                    if state in policy_to_plot.keys():
                        if label==red:
                            if fixed_idx: # Env is fixed
                                if (state[robot_idx] in fixed_obstacle_cells) or (state[robot_idx] == state[env_idx]):
                                    grid_row, grid_col = np.where(grid_map==state[robot_idx])
                                    this_maze[grid_row, grid_col] = cmap.colors.index('red')
                        if label==green:
                            grid_row, grid_col = np.where(grid_map==state[robot_idx])
                            this_maze[grid_row, grid_col] = cmap.colors.index('green')
                grid_row, grid_col = np.where(grid_map==pose)
                this_maze[grid_row, grid_col] = cmap.colors.index('blue')
                base_policy_grid.updateCellColors(maze_cells=this_maze)

                fig = base_policy_grid.configurePlot(title, policy_to_plot, act_list, use_print_keys,
                                                     plot_key_dict[pose], decimals=2,
                                                     kernel_locations=kernel_loc, stay_action=act_list[0])

        print '\n\nHEY! You! With the face! (computers don\'t have faces) Mazimize figure window to correctly show ' \
                'arrow/dot size ratio!\n'

    if plot_uncertainty:
        # Only for GaussianTheta
        uncertainty_grid = PlotHelper.UncertaintyPlot(maze, cmap, grid_map)
        policy_uncertainty = infer_mdp.policy_uncertainty_as_vec.reshape([infer_mdp.num_states, infer_mdp.num_actions])
        for act_idx, act in enumerate(action_list):
            param_vector_indeces = xrange(act_idx, len(infer_mdp.theta), len(action_list))
            uncertainty_vals = infer_mdp.theta_std_dev[param_vector_indeces]
            title='Param Uncertainty'
            fig, ax = uncertainty_grid.configurePlot(title, infer_mdp.kernel_centers, uncertainty_vals, act_str=str(act))
            # Plot aggregate uncertainty at states here
            fig, ax = uncertainty_grid.configurePlot('Policy Uncertainty', infer_mdp.grid_map.ravel(),
                                                     policy_uncertainty[:, act_idx], act_str=str(act))


    plt.show()

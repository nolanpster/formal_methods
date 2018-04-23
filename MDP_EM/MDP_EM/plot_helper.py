#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'


from copy import deepcopy
import numpy as np
from pprint import pprint
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D

# Set warning filter before pandas import so pandas recognizes it.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

# Module dictionary used to select which plots to make.
default_plot_flags = {'VI': False,
                      'EM': False,
                      'inference': False,
                      'inference_statistics': False,
                      'demonstration': False,
                      'phi': False,
                      'phi_std_dev': False,
                      'bonus_reward': False}

# Module Dictionary used to pass arguments around.
default_plot_info = dict.fromkeys(('fixed_obs_labels', 'grid_map', 'labels', 'alphabet_dict', 'num_agents',
                                   'robot_goal_states', 'robot_action_list', 'env_action_list'))

class PlotGrid(object):

    @staticmethod
    def buildGridPlotArgs(grid_map, labels, alphabet_dict, num_agents=1, agent_idx=0, fixed_obstacle_labels=None,
                          goal_states=None, labels_have_dra_states=False):
        """
        @param fixed_obstacle_labels is required if num_agents > 1.
        @param if goals states are provided, the goals cell value should correspond to the agent index.
        """
        if labels_have_dra_states:
            # Expect states to be stored as (grid_cell_state, dra_state) or, (grid_cell_state,)
            cell_state_slicer = slice(0,1)
        else:
            # Expect states to be stored as (grid_cell_state)
            cell_state_slicer = slice(None)
        grid_dim = grid_map.shape
        # Build the color list in order of numerical values in maze.
        color_list = ['white', 'green']
        if alphabet_dict['red'] in labels.values():
            # Next highest value in maze corresponds to red.
            color_list.append('red')
        if num_agents > 1:
            # Next highest value in maze corresponds to blue.
            color_list.append('blue')
        # Create plots for comparison. Note that the the `maze` array has one more row and column than the `grid` for
        # plotting purposes.
        maze = np.zeros(np.array(grid_dim)+1)
        for state, label in labels.iteritems():
            if label==alphabet_dict['red']:
                if num_agents > 1 and agent_idx > 0:
                    if fixed_obstacle_labels[state[cell_state_slicer][0]]==alphabet_dict['red']:
                        # Assume fixed obstacles are shared, only grab the robot state.
                        grid_row, grid_col = np.where(grid_map==state[cell_state_slicer][0][0])
                        maze[grid_row, grid_col] = color_list.index('red')
                else:
                    if num_agents > 1:
                        if fixed_obstacle_labels[state[cell_state_slicer]]==alphabet_dict['red']:
                            grid_row, grid_col = np.where(grid_map==state[cell_state_slicer][agent_idx])
                            maze[grid_row, grid_col] = color_list.index('red')
                    else:
                        grid_row, grid_col = np.where(grid_map==state[cell_state_slicer])
                        maze[grid_row, grid_col] = color_list.index('red')
            if label==alphabet_dict['green'] and goal_states is None:
                if num_agents > 1:
                    grid_row, grid_col = np.where(grid_map==state[cell_state_slicer][0])
                else:
                    grid_row, grid_col = np.where(grid_map==state[cell_state_slicer])
                maze[grid_row, grid_col] = color_list.index('green')
            elif goal_states is not None and state in goal_states:
                if num_agents > 1:
                    grid_row, grid_col = np.where(grid_map==state[cell_state_slicer][agent_idx])
                else:
                    grid_row, grid_col = np.where(grid_map==state[cell_state_slicer])
                maze[grid_row, grid_col] = color_list.index('green')

        cmap = mcolors.ListedColormap(color_list)
        return maze, cmap

    def __init__(self, maze_cells, cmap, fontsize=20):
        # maze_cells - np.array - grid of ints, 0 is colored with cell_color[0], 1 is colored with cell_color[1], etc.
        #              expectes one EXTRA row and column for formatting.
        # cmap - color map providing color_cell list above (type mcolors.ListedColors())
        self.maze_cells = maze_cells
        self.grid_dim = maze_cells.shape
        self.x, self.y = np.meshgrid(np.arange(self.grid_dim[1]), np.arange(self.grid_dim[0]))
        self.cmap = cmap
        self.fontsize = fontsize

    def configurePlot(self, title):
        fig = plt.figure(figsize=(16.5, 13), dpi=50)
        ax = fig.add_subplot(1, 1, 1)
        self.quadmesh = ax.pcolormesh(self.x, self.y, self.maze_cells, edgecolor='k', linewidth=0.5, cmap=self.cmap)
        ax.set_title(title, fontsize=self.fontsize)
        return fig, ax


class PlotKernel(PlotGrid):
    """
    @brief Can be used to plot a Kernel or the Phi values.
    """
    def __init__(self, maze_cells, cmap, action_list, grid_map):
        # Drop last row and column from maze_cells due to formatting decision for super class.
        super(self.__class__, self).__init__(maze_cells[:-1, :-1], cmap)
        self.action_list = action_list
        self.grid_map = grid_map

    def configurePlot(self, title, elem_idx, kernels=None, phi_at_state=None, act=None, states=None, do_print=False):
        """
        @param Title
        @param elem_idx The index of the kernel vector, or phi_at_state to print; e.g., if there are kernels at cells
               [0, 2, 3] then elem_idx=1 will access the kernel (or coresponding phi values) centered at grid-cell 2.
        @param act Used for plotting phi values (they are a function the action taken).
        @param states a list of states to use as keys in phi_at_state. If `None`, cell indeces in PlotKernel.GridMap
               will be used.
        """
        fig, ax = super(self.__class__, self).configurePlot(title)
        if phi_at_state is not None:
            if states is None:
                states = range(self.grid_map.size)
            try:
                bar_height = np.array([phi_at_state[state][act]
                                       [len(self.action_list)*(elem_idx) + self.action_list.index(act)] for state in
                                       states]).reshape(self.grid_dim)
            except:
                raise KeyError('Invalid state-key in the phi_at_state dictionary.')
        elif kernels is not None:
            bar_height = np.array([kernels[elem_idx](state)
                            for state in range(self.grid_map.size)]).reshape(self.grid_dim)
        else:
            raise ValueError('No input values to plot!')
        if do_print:
            print('Values of bars in {} plot.'.format('kernels' if kernels is not None else 'phi'))
            pprint(bar_height)
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(elev=46, azim=-76)

        num_cells = bar_height.size
        zpos = np.zeros(num_cells)
        dx = np.ones(num_cells)
        dy = np.ones(num_cells)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax1.bar3d(self.x.ravel(), self.y.ravel(), zpos, dx, dy, bar_height.ravel(), color='#00ceaa')
            # Invert y-axis because we're plotting this like an image with origin in upper left corner.
            ax1.invert_yaxis()
        plt.axis('off')
        plt.title(str(act))
        #plt.savefig('elem_idx_0_act_{}.tif'.format(act), dpi=400, transparent=False)
        fig.tight_layout()
        return fig, ax1

class UncertaintyPlot(PlotGrid):

    def __init__(self, maze_cells, cmap, grid_map):
        # Drop last row and column from maze_cells due to formatting decision for super class.
        super(self.__class__, self).__init__(maze_cells[:-1, :-1], cmap)
        self.grid_map = grid_map

    def configurePlot(self, title, uncertainty_locations, uncertainty_magnitude, only_param_values=True, act_str=None,
                      do_print=False):
        """
        """
        fig, ax = super(self.__class__, self).configurePlot(title)
        bar_height = np.zeros(self.maze_cells.size)
        for cell, mag_val in zip(uncertainty_locations, uncertainty_magnitude):
            bar_height[cell] = mag_val
        bar_height.reshape(self.grid_dim)

        if do_print:
            print('Values of bars in {} uncertainty plot.'.format('parameter' if only_param_values else ''))
            pprint(bar_height)
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(elev=46, azim=-76)

        num_cells = bar_height.size
        zpos = np.zeros(num_cells)
        dx = np.ones(num_cells)
        dy = np.ones(num_cells)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax1.bar3d(self.x.ravel(), self.y.ravel(), zpos, dx, dy, bar_height.ravel(), color='#00ceaa')
            # Invert y-axis because we're plotting this like an image with origin in upper left corner.
            ax1.invert_yaxis()
        plt.axis('off')
        plt.title(title+act_str, fontsize=self.fontsize)
        #plt.savefig('elem_idx_0_act_{}.tif'.format(act_str), dpi=400, transparent=False)
        fig.tight_layout()
        return fig, ax1

class PlotPolicy(PlotGrid):

    def __init__(self, maze_cells, cmap, center_offset):
        super(self.__class__, self).__init__(maze_cells, cmap)
        self.x_cent = self.x[:-1,:-1].ravel()+center_offset
        self.y_cent = self.y[:-1,:-1].ravel()+center_offset
        self.zero_mag = np.zeros(np.array(self.grid_dim))
        # Make these configurable eventually.
        self.quiv_angs = {'North': np.pi/2, 'South': -np.pi/2, 'East': 0, 'West': np.pi}
        self.quiv_scale = 15
        self.stay_scale = 250
        self.prob_disp_thresh = 0.02
        self.predefined_action_set = frozenset(self.quiv_angs.keys())

    def configurePlot(self, title, policy, action_list, use_print_keys=False, policy_keys_to_print=None, decimals=2,
                       kernel_locations=None, stay_action='Empty', do_print=False):
        fig, ax = super(self.__class__, self).configurePlot(title)
        if not any(frozenset(action_list) & self.predefined_action_set):
            self.setQuivActions(action_list)
        policy = deepcopy(policy)
        # Stay probabilies - plot with dots.
        if use_print_keys:
            stay_probs = [np.round(policy[state][stay_action], decimals) for state in policy_keys_to_print]
        else:
            stay_probs = [np.round(policy[state][stay_action][0], decimals) for state in policy.keys()]
        # Deepcopy allows for each action not to plot cells when there is probability of the action rounds to zero.
        this_x_cent = deepcopy(self.x_cent)
        this_y_cent = deepcopy(self.y_cent)
        this_x_cent = [x for idx,x in enumerate(this_x_cent) if stay_probs[idx] > self.prob_disp_thresh]
        this_y_cent = [y for idx,y in enumerate(this_y_cent) if stay_probs[idx] > self.prob_disp_thresh]
        try:
            if type(stay_probs).__module__=='numpy':
                stay_probs = stay_probs.tolist()
        except SyntaxError:
            pass
        stay_probs = [p for p in stay_probs if  p > self.prob_disp_thresh]
        if any(stay_probs):
            df_stay = pd.DataFrame({'Prob': stay_probs, 'x': this_x_cent, 'y': this_y_cent})
            df_stay.plot(kind='scatter', x='x', y='y', s=df_stay['Prob']*self.stay_scale, c=1-df_stay['Prob'], ax=ax,
                         cmap='gray', legend=None)
        # Motion actions - plot with arrows.
        for act in action_list:
            if act==stay_action:
                continue # Alraedy plotted
            # Deepcopy allows for each action not to plot cells when there is probability of the action rounds to zero.
            this_x_cent = deepcopy(self.x_cent)
            this_y_cent = deepcopy(self.y_cent)
            if use_print_keys:
                act_probs = np.round([policy[state][act] for state in policy_keys_to_print], decimals)
            else:
                act_probs = np.round([policy[state][act][0] for state in policy.keys()], decimals)
            this_x_cent = [x for idx,x in enumerate(this_x_cent) if act_probs[idx] > self.prob_disp_thresh]
            this_y_cent = [y for idx,y in enumerate(this_y_cent) if act_probs[idx] > self.prob_disp_thresh]
            act_probs = [p for p in act_probs if  p > self.prob_disp_thresh]
            act_probs = np.array(act_probs) # This is redundanct for everything except the value iteration policy.
            U = np.cos(self.quiv_angs[act])*act_probs
            V = np.sin(self.quiv_angs[act])*act_probs
            Q = plt.quiver(this_x_cent, this_y_cent, U, V, scale=self.quiv_scale, units='width')

        if kernel_locations is not None:
            # Add marker to cells at kernel centers.
            circ_handles = []
            radius = 0.5 # Relative to cell width.
            for kern_cell in kernel_locations:
                circ_handles += [Wedge((self.x_cent[kern_cell], self.y_cent[kern_cell]), radius, 0, 360, width=0.05)]
            circle_collection = PatchCollection(circ_handles, alpha=0.4)
            ax.add_collection(circle_collection)

        plt.gca().invert_yaxis()
        fig.tight_layout()
        return fig

    def setQuivActions(self, action_list):
        for act in action_list:
            for preset_action in self.predefined_action_set:
                if preset_action in act:
                    self.quiv_angs[act] = self.quiv_angs[preset_action]

    def updateCellColors(self, maze_cells=None, cmap=None):
        if maze_cells is not None:
            self.maze_cells = maze_cells
            self.grid_dim = maze_cells.shape
            self.x, self.y = np.meshgrid(np.arange(self.grid_dim[1]), np.arange(self.grid_dim[0]))
        if cmap is not None:
            self.cmap = cmap

class PlotDemonstration(PlotGrid):

    def __init__(self, maze_cells, cmap, center_offset):
        super(self.__class__, self).__init__(maze_cells, cmap)
        self.x_cent = self.x[:-1,:-1].ravel()+center_offset
        self.y_cent = self.y[:-1,:-1].ravel()+center_offset

    def configurePlot(self, title, histories, do_print=False):

        fig, ax = super(self.__class__, self).configurePlot(title)
        cells_visited, times_in_demo = np.unique(histories, return_counts=True)
        this_x_cent = deepcopy(self.x_cent)
        this_y_cent = deepcopy(self.y_cent)
        this_x_cent = [x for idx,x in enumerate(this_x_cent) if idx in cells_visited]
        this_y_cent = [y for idx,y in enumerate(this_y_cent) if idx in cells_visited]

        for x, y, count in zip(this_x_cent, this_y_cent, times_in_demo):
            plt.text(x, y, str(count), fontsize=self.fontsize)
        plt.gca().invert_yaxis()
        fig.tight_layout()

def plotPolicyErrorVsNumberOfKernels(kernel_set_L1_err, number_of_kernels_in_set, title, mle_L1_norm=None):
    """
    @param kernel_set_L1_err A [NxM] numpy array where N is the number of kernel sets used and M is the number of
           trials at for each set.
    @param number_of_kernels_in_set A numpy array of how many kernels are in each set, length N.

    @note X-limits are automatically set and assume a constant interval between the number of kernels in each set.
    """
    fig = plt.figure(figsize=(13.0, 5), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    kernel_count_min = number_of_kernels_in_set.min()
    kernel_count_max = number_of_kernels_in_set.max()
    kernel_count_incr = (kernel_count_max - kernel_count_min) / len(number_of_kernels_in_set)
    plt.xlim((kernel_count_min - kernel_count_incr, kernel_count_max + kernel_count_incr))
    means = kernel_set_L1_err.mean(axis=1)
    stds = kernel_set_L1_err.std(axis=1)
    mins = kernel_set_L1_err.min(axis=1)
    maxes = kernel_set_L1_err.max(axis=1)
    plt.errorbar(x=number_of_kernels_in_set, y=means, yerr=stds, fmt='ok', lw=3)
    plt.errorbar(x=number_of_kernels_in_set, y=means, yerr= [means - mins, maxes - means], fmt='.k', ecolor='gray',
                 lw=1)
    #plt.title(title)
    #plt.ylabel('L1-Norm Error')
    #plt.xlabel('Kernel Count')
    ax.set_xticks(number_of_kernels_in_set)
    if mle_L1_norm is not None:
        ax.axhline(y=mle_L1_norm, color='navy', linestyle='--', linewidth=3)
    #plt.savefig('error_bars_large_skinny.tif', dpi=400, transparent=False)

    fig.tight_layout()
    return fig, ax

def plotValueStatsVsBatch(val_array_1, title='L1-Norm', ylabel='Fraction of Max', xlabel='Batch', data_label_1='Active',
                          color_1='k', val_array_2=None, data_label_2='Passive', color_2='r', transparency=0.3,
                          plot_quantiles=True, plot_min_max=False):

    if val_array_2 is not None and (val_array_1.shape != val_array_2.shape):
        raise ValueError('Input data, value array 1 & value array 2, are not the same size!')
    #fig = plt.figure(figsize=(13.0, 5), dpi=100)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    num_batches, num_trials = val_array_1.shape
    batch_count = range(num_batches)

    mean_1 = plt.plot(np.mean(val_array_1, axis=1), label=data_label_1, color=color_1)
    mean_2 = plt.plot(np.mean(val_array_2, axis=1), label=data_label_2, color=color_2)

    if plot_quantiles:
        val_1_quant25, val_1_quant75 = np.percentile(val_array_1, [25, 75], axis=1)
        plt.fill_between(batch_count, y1=val_1_quant75, y2=val_1_quant25, color=color_1, alpha=transparency,
                         linestyle='--')
        val_2_quant25, val_2_quant75 = np.percentile(val_array_2, [25, 75], axis=1)
        plt.fill_between(batch_count, y1=val_2_quant75, y2=val_2_quant25, color=color_2, alpha=transparency,
                         linestyle='--')

        # Hack for legend of region
        plt.plot([], [], linewidth=10, color=color_1, alpha=transparency, label=data_label_1 + " 25%-75%")
        plt.plot([], [], linewidth=10, color=color_2, alpha=transparency, label=data_label_2 + " 25%-75%")

    if plot_min_max:
        if plot_quantiles:
            this_transparency = transparency / 2
        else:
            this_transparency = transparency

        plt.fill_between(batch_count, y1=np.max(val_array_1, axis=1), y2=np.min(val_array_1, axis=1), color=color_1,
                         alpha=this_transparency, linestyle=':')
        plt.fill_between(batch_count, y1=np.max(val_array_2, axis=1), y2=np.min(val_array_2, axis=1), color=color_2,
                         alpha=this_transparency, linestyle=':')

        # Hack for legend of min/max region
        plt.plot([], [], linewidth=10, color=color_1, alpha=this_transparency, label=data_label_1 + " Min/Max")
        plt.plot([], [], linewidth=10, color=color_2, alpha=this_transparency, label=data_label_2 + " Min/Max")

    plt.title(title + " - {} Trials".format(num_trials))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    ax.set_xticks(batch_count)
    plt.xlim(xmax=num_batches)
    plt.legend()
    fig.tight_layout()

def plotValueVsBatch(val_array, title, ylabel, xlabel='Batch', also_plot_stats=False, stats_label='',
                     save_figures=False):
    """
    @note X-limits are automatically set and assume a constant interval between the number of kernels in each set.
    """
    #fig = plt.figure(figsize=(13.0, 5), dpi=100)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    num_batches, num_trials = val_array.shape
    batch_count = range(num_batches)

    plt.plot(val_array)

    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    ax.set_xticks(batch_count)
    plt.xlim(xmax=num_batches)
    plt.legend()
    fig.tight_layout()

    if save_figures:
        plt.savefig('values_{}_{}trials_{}batches.tif'.format(title, num_trials, num_batches, dpi=400,
                    transparent=False))

    if also_plot_stats:
        #stats_fig = plt.figure(figsize=(13.0, 5), dpi=100)
        stats_fig = plt.figure()
        stats_ax = stats_fig.add_subplot(1, 1, 1)

        means = val_array.mean(axis=1)
        stds = val_array.std(axis=1)
        mins = val_array.min(axis=1)
        maxes = val_array.max(axis=1)
        plt.errorbar(x=batch_count, y=means, yerr=stds, fmt='ok', lw=3, label=stats_label+' Mean & Std. Dev.')
        plt.errorbar(x=batch_count, y=means, yerr= [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1)

        if title:
            plt.title(title)
        if ylabel:
            plt.ylabel(ylabel)
        if xlabel:
            plt.xlabel(xlabel)
        stats_ax.set_xticks(batch_count)
        plt.xlim(xmax=num_batches)
        plt.legend()
        fig.tight_layout()

        if save_figures:
            plt.savefig('stats_{}_{}trials_{}batches.tif'.format(title, num_trials, num_batches, dpi=400,
                        transparent=False))

    return_tuple = (fig, ax) if not also_plot_stats else (fig, ax, stats_fig, stats_ax)
    return return_tuple

def makePlotGroups(plot_all_grids=False, plot_VI_mdp_grids=False, plot_EM_mdp_grids=False,
                   plot_inferred_mdp_grids=False, VI_mdp=None, EM_mdp=None, infer_mdp=None, robot_action_list=None,
                   env_action_list=None, VI_plot_keys=None, EM_plot_keys=None, infer_plot_keys=None,
                   include_kernels=False):

    mdp_list = []
    plot_policies = []
    only_use_print_keys = []
    titles = []
    kernel_locations = []
    action_lists = []
    plot_key_groups = []
    if plot_all_grids or (plot_VI_mdp_grids and plot_EM_mdp_grids and plot_inferred_mdp_grids):
        mdp_list = [VI_mdp, EM_mdp, infer_mdp]
        plot_policies = [VI_mdp.policy, EM_mdp.policy, infer_mdp.policy]
        titles = ['Value Iteration', 'Expecation Maximization', 'Learned']
        only_use_print_keys = [True, True, False]
        if include_kernels:
            kernel_locations = [None, None, infer_mdp.kernel_centers]
        action_lists = [robot_action_list, robot_action_list, env_action_list]
        plot_key_groups = [VI_plot_keys, EM_plot_keys, infer_plot_keys]
        return mdp_list, plot_policies, only_use_print_keys, titles, kernel_locations, action_lists, plot_key_groups

    if plot_VI_mdp_grids:
        mdp_list.append(VI_mdp)
        plot_policies.append(VI_mdp.policy)
        titles.append('Value Iteration')
        if include_kernels:
            kernel_locations.append(None)
        only_use_print_keys.append(True)
        action_lists.append(robot_action_list)
        plot_key_groups.append(VI_plot_keys)

    if plot_EM_mdp_grids:
        mdp_list.append(EM_mdp)
        plot_policies.append(EM_mdp.policy)
        titles.append('Expectation Maximization')
        if include_kernels:
            kernel_locations.append(None)
        only_use_print_keys.append(True)
        action_lists.append(robot_action_list)
        plot_key_groups.append(EM_plot_keys)

    if plot_inferred_mdp_grids:
        mdp_list.append(infer_mdp)
        plot_policies.append(infer_mdp.policy)
        titles.append('Learned')
        only_use_print_keys.append(False)
        if include_kernels:
            kernel_locations.append(infer_mdp.kernel_centers)
        action_lists.append(env_action_list)
        plot_key_groups.append(infer_plot_keys)

    return mdp_list, plot_policies, only_use_print_keys, titles, kernel_locations, action_lists, plot_key_groups

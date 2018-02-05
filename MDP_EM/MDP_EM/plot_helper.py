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
import pandas as pd

class PlotGrid(object):
    def __init__(self, maze_cells, cmap):
        # maze_cells - np.array - grid of ints, 0 is colored with cell_color[0], 1 is colored with cell_color[1], etc.
        #              expectes one EXTRA row and column for formatting.
        # cmap - color map providing color_cell list above (type mcolors.ListedColors())
        self.maze_cells = maze_cells
        self.grid_dim = maze_cells.shape
        self.x, self.y = np.meshgrid(np.arange(self.grid_dim[1]), np.arange(self.grid_dim[0]))
        self.cmap = cmap

    def configurePlot(self, title):
        fig, ax = plt.subplots()
        self.quadmesh = ax.pcolormesh(self.x, self.y, self.maze_cells, edgecolor='k', cmap=self.cmap)
        plt.title(title)
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

    def configurePlot(self, title, cell, kernels=None, phi_at_state=None, act=None):
        """
        @param Title
        @param cell The index of the kernel vector, or phi_at_state to print; e.g., if there are kernels at cells
               [0, 2, 3] then cell=1 will access the kernel (or coresponding phi values) centered at grid-cell 2.
        @param action Used for plotting phi values.
        """
        fig, ax = super(self.__class__, self).configurePlot(title)
        if phi_at_state is not None:
            try:
                bar_height = np.array([phi_at_state[state][act][len(self.action_list)*(cell) + self.action_list.index(act)] for state
                    in range(self.grid_map.size)]).reshape(self.grid_dim)
            except:
                # Determine type of error to raise when cell is invalid.
                import pdb; pdb.set_trace()
        elif kernels is not None:
            bar_height = np.array([kernels[cell](state) for state in range(self.grid_map.size)]).reshape(self.grid_dim)
        else:
            raise ValueError('No input values to plot!')
        print('Values of bars in {} plot.'.format('kernels' if kernels is not None else 'phi'))
        pprint(bar_height)
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(elev=56, azim=-31)

        num_cells = bar_height.size
        zpos = np.zeros(num_cells)
        dx = np.ones(num_cells)
        dy = np.ones(num_cells)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax1.bar3d(self.x.ravel(), self.y.ravel(), zpos, dx, dy, bar_height.ravel(), color='#00ceaa')
            # Invert y-axis because we're plotting this like an image with origin in upper left corner.
            ax1.invert_yaxis()
        return fig, ax1

class PlotPolicy(PlotGrid):

    def __init__(self, maze_cells, cmap, center_offset):
        super(self.__class__, self).__init__(maze_cells, cmap)
        self.x_cent = self.x[:-1,:-1].ravel()+center_offset
        self.y_cent = self.y[:-1,:-1].ravel()+center_offset
        self.zero_mag = np.zeros(np.array(self.grid_dim))
        # Make these configurable eventually.
        self.quiv_angs = {'North': np.pi/2, 'South': -np.pi/2, 'East': 0, 'West': np.pi}
        self.quiv_scale = 20
        self.stay_scale = 250
        self.prob_disp_thresh = 0.02

    def configurePlot(self, title, policy, action_list, use_print_keys, policy_keys_to_print, decimals,
                       kernel_locations=None):
        fig, ax = super(self.__class__, self).configurePlot(title)
        policy = deepcopy(policy)
        # Stay probabilies - plot with dots.
        if use_print_keys:
            stay_probs = [np.round(policy[state]['Empty'], decimals) for state in policy_keys_to_print]
        else:
            stay_probs = [np.round(policy[state]['Empty'][0][0], decimals) for state in policy.keys()]
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
        df_stay = pd.DataFrame({'Prob': stay_probs, 'x': this_x_cent, 'y': this_y_cent})
        df_stay.plot(kind='scatter', x='x', y='y', s=df_stay['Prob']*self.stay_scale, c=1-df_stay['Prob'], ax=ax,
                     cmap='gray', legend=None)
        # Motion actions - plot with arrows.
        for act in action_list:
            if act=='Empty':
                continue # Alraedy plotted
            # Deepcopy allows for each action not to plot cells when there is probability of the action rounds to zero.
            this_x_cent = deepcopy(self.x_cent)
            this_y_cent = deepcopy(self.y_cent)
            if use_print_keys:
                act_probs = np.round([policy[state][act] for state in policy_keys_to_print], decimals)
            else:
                act_probs = np.round([policy[state][act][0][0] for state in policy.keys()], decimals)
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
        return fig


def plotPolicyErrorVsNumberOfKernels(kernel_set_L1_err, number_of_kernels_in_set, title):
    """
    @param kernel_set_L1_err A [NxM] numpy array where N is the number of kernel sets used and M is the number of
           trials at for each set.
    @param number_of_kernels_in_set A numpy array of how many kernels are in each set, length N.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    means = kernel_set_L1_err.mean(axis=1)
    stds = kernel_set_L1_err.std(axis=1)
    mins = kernel_set_L1_err.min(axis=1)
    maxes = kernel_set_L1_err.max(axis=1)
    plt.errorbar(x=number_of_kernels_in_set, y=means, yerr=stds, fmt='ok', lw=3)
    plt.errorbar(x=number_of_kernels_in_set, y=means, yerr= [means - mins, maxes - means], fmt='.k', ecolor='gray',
                 lw=1)
    plt.title(title)
    plt.ylabel('L1-Norm Error')
    plt.xlabel('Kernel Count')
    ax.set_xticks(number_of_kernels_in_set)

    return fig, ax

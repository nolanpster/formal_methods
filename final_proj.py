#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'


from NFA_DFA_Module.DFA import DRA
from NFA_DFA_Module.DFA import LTL_plus
from MDP_EM.MDP_EM.MDP import MDP
from MDP_EM.MDP_EM.grid_graph import GridGraph

import os
import datetime
import time
import pickle
import dill # For pickling lambda functions.
import csv
import numpy as np
from numpy import ma
from copy import deepcopy
from pprint import pprint
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import warnings

np.set_printoptions(linewidth=300)
np.set_printoptions(precision=3)

mdp_obj_path = os.path.abspath('pickled_mdps')
data_path = os.path.abspath('pickled_episodes')
infered_mdps_path = os.path.abspath('pickled_inference')

def getOutFile(name_prefix='EM_MDP', dir_path=mdp_obj_path):
    # Dev machine returns UTC.
    current_datetime = datetime.datetime.now()
    formatted_time = current_datetime.strftime('_UTC%y%m%d_%H%M')
    # Filepath for mdp objects.
    full_file_path = os.path.join(dir_path, name_prefix + formatted_time)
    if not os.path.exists(os.path.dirname(full_file_path)):
        try:
            os.makedirs(os.path.dirname(full_file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return full_file_path

# Transition probabilities for each action in each cell (gross, explodes with
# the number of states).
#
# Row transition probabilites are:
# 0) Transition prob from Normal Grid cell to another normal grid cell.
# 1) Transition prob when adjacent to North Wall
# 2) Transition prob when adjacent to South Wall
# 3) Transition prob when adjacent to East Wall
# 4) Transition prob when adjacent to West Wall
# 5) Transition prob when in NE corner cell.
# 6) Transition prob when in NW corner cell.
# 7) Transition prob when in SE corner cell.
# 8) Transition prob when in SW corner cell.
#
# Column values are probabilities of ['Empty', 'north', 'south', 'east', 'west'] actions.
act_prob = {'North': np.array([[0.0, 0.8, 0.0, 0.1, 0.1],
                               [0.8, 0.0, 0.0, 0.1, 0.1],
                               [0.0, 0.8, 0.0, 0.1, 0.1],
                               [0.1, 0.8, 0.0, 0.0, 0.1],
                               [0.1, 0.8, 0.0, 0.1, 0.0],
                               [0.9, 0.0, 0.0, 0.0, 0.1],
                               [0.9, 0.0, 0.0, 0.1, 0.0],
                               [0.1, 0.8, 0.0, 0.0, 0.1],
                               [0.1, 0.8, 0.0, 0.1, 0.0]]
                               ),
            'South': np.array([[0.0, 0.0, 0.8, 0.1, 0.1],
                               [0.0, 0.0, 0.8, 0.1, 0.1],
                               [0.8, 0.0, 0.0, 0.1, 0.1],
                               [0.1, 0.0, 0.8, 0.0, 0.1],
                               [0.1, 0.0, 0.8, 0.1, 0.0],
                               [0.1, 0.0, 0.8, 0.0, 0.1],
                               [0.1, 0.0, 0.8, 0.1, 0.0],
                               [0.9, 0.0, 0.0, 0.0, 0.1],
                               [0.9, 0.0, 0.0, 0.1, 0.0]]
                               ),
            'East': np.array([[0.0, 0.1, 0.1, 0.8, 0.0],
                              [0.1, 0.0, 0.1, 0.8, 0.0],
                              [0.1, 0.1, 0.0, 0.8, 0.0],
                              [0.8, 0.1, 0.1, 0.0, 0.0],
                              [0.0, 0.1, 0.1, 0.8, 0.0],
                              [0.9, 0.0, 0.1, 0.0, 0.0],
                              [0.1, 0.0, 0.1, 0.8, 0.0],
                              [0.9, 0.1, 0.0, 0.0, 0.0],
                              [0.1, 0.1, 0.0, 0.8, 0.0]]
                              ),
            'West': np.array([[0.0, 0.1, 0.1, 0.0, 0.8],
                              [0.1, 0.0, 0.1, 0.0, 0.8],
                              [0.1, 0.1, 0.0, 0.0, 0.8],
                              [0.0, 0.1, 0.1, 0.0, 0.8],
                              [0.8, 0.1, 0.1, 0.0, 0.0],
                              [0.1, 0.0, 0.1, 0.0, 0.8],
                              [0.9, 0.0, 0.1, 0.0, 0.0],
                              [0.1, 0.1, 0.0, 0.0, 0.8],
                              [0.9, 0.1, 0.0, 0.0, 0.0]]
                              ),
            'Empty': np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0]]
                               )
            }

infer_act_prob = {'North': np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0]]
                                    ),
                 'South': np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0]]
                                    ),
                 'East': np.array([[0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0]]
                                   ),
                 'West': np.array([[0.0, 0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0, 0.0, 0.0]]
                                   ),
                 'Empty': np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0, 0.0]]
                                    )
                 }

grid_dim = [16, 8] # [num-rows, num-cols]
grid_map = np.array(range(0,np.prod(grid_dim)), dtype=np.int8).reshape(grid_dim)
states = [str(state) for state in range(grid_map.size)]

# Shared MDP Initialization Parameters.
green = LTL_plus('green')
red = LTL_plus('red')
empty = LTL_plus('E')
# Where empty is the empty string/label/dfa-action.
atom_prop = [green, red, empty]
# Defined empty action for MDP incurrs a self loop.
action_list = ['Empty', 'North', 'South', 'East', 'West']
# Set `solve_with_uniform_distribution` to True to have the initial distribution for EM and the history/demonstration
# generation start with a uniform (MDP.S default) distribution across the values assigned to MDP.init_set. Set this to
# _False_ to have EM and the MDP always start from the `initial_state` below.
solve_with_uniform_distribution = False
initial_state = '127'
labels = {state: empty for state in states}
labels['36'] = red
labels['37'] = red
labels['44'] = red
labels['45'] = red
labels['18'] = green
goal_state = 18 # Currently assumess only one goal.


def makeGridMDPxDRA(do_print=False):
    ##### Problem 2 - Configure MDP #####
    # For the simple 6-state gridworld, see slide 8 of Lecture 7, write the specification automata for the following:
    # visit all green cells and avoid the red cell.
    #
    # Note shared atomic propositions:
    # Note that input gamma is overwritten in DRA/MDP product method, so we'll need to set it again later.
    grid_mdp = MDP(init=initial_state, action_list=action_list, states=states, act_prob=deepcopy(act_prob), gamma=0.9,
                   AP=atom_prop, L=labels, grid_map=grid_map)
    if solve_with_uniform_distribution:
        # Leave MDP.init_set unset (=None) if you want to solve the system from a single initial_state.
        grid_mdp.init_set = grid_mdp.states

    ##### Add DRA for co-safe spec #####
    # Define a Deterministic (finitie) Raban Automata to match the sketch on slide 7 of lecture 8. Note that state 'q2'
    # is is the red, 'sink' state.
    co_safe_dra = DRA(initial_state='q0', alphabet=atom_prop, rabin_acc=[({'q1'},{})])
    # Self-loops = Empty transitions
    co_safe_dra.add_transition(empty, 'q0', 'q0') # Initial state
    co_safe_dra.add_transition(empty, 'q2', 'q2') # Losing sink.
    co_safe_dra.add_transition(empty, 'q3', 'q3') # Winning sink.
    # Labeled transitions
    co_safe_dra.add_transition(green, 'q0', 'q1')
    co_safe_dra.add_transition(green, 'q1', 'q1')
    co_safe_dra.add_transition(red, 'q0', 'q2')
    # If the DRA reaches state 'q1' we win. Therefore I do not define a transition from 'q3' to 'q4'. Note that 'q4' is
    # a sink state due to the self loop.
    #
    # Also, I define a winning 'sink' state, 'q3'. I do this so that there is only one out-going transition from 'q1'
    # and it's taken only under the empty action. This action, is the winning action. This is a little bit of a hack,
    # but it was the way that I thought of to prevent the system from repeatedly taking actions that earned a reward.
    co_safe_dra.add_transition(empty, 'q1', 'q3')
    # Not adding a transition from 'q3' to 'q2' under red for simplicity. If we get to 'q3' we win.
    if False:
        co_safe_dra.toDot('visitGreensAndNoRed.dot')
        pprint(vars(co_safe_dra))
    VI_game_mdp = MDP.productMDP(grid_mdp, co_safe_dra)
    VI_game_mdp.grid_map = grid_map
    # Define the reward function for the VI_game_mdp. Get a reward when leaving
    # the winning state 'q1' to 'q3'.
    pos_reward = {
                 'North': 0.0,
                 'South': 0.0,
                 'East': 0.0,
                 'West': 0.0,
                 'Empty': 1.0
                 }
    no_reward = {
                 'North': 0.0,
                 'South': 0.0,
                 'East': 0.0,
                 'West': 0.0,
                 'Empty': 0.0
                 }
    # Go through each state and if it is a winning state, assign it's reward
    # to be the positive reward dictionary. I have to remove the state
    # ('5', 'q3') because there are conflicting actions due to the label of '4'
    # being 'red'.
    reward_dict = {}
    for state in VI_game_mdp.states:
        if state in VI_game_mdp.acc[0][0] and not VI_game_mdp.L[state]==red:
            # Winning state
            reward_dict[state] = pos_reward
        else:
            # No reward when leaving current state.
            reward_dict[state] = no_reward
    VI_game_mdp.reward = reward_dict
    # Then I set up all sink states so all transition probabilities from a sink
    # states take a self loop with probability 1.
    VI_game_mdp.sink_act = 'Empty'
    VI_game_mdp.setSinks('q3')
    # If I uncomment the following line, all states at grid cell '5' no longer
    # build up any reward.
    #VI_game_mdp.setSinks('5')
    VI_game_mdp.setSinks('q2')
    # @TODO Prune unreachable states from MDP.
    EM_game_mdp = deepcopy(VI_game_mdp)
    EM_game_mdp.setInitialProbDist(EM_game_mdp.init_set)

    # Compare policy (action probabilities).
    policy_keys_to_print = deepcopy([(state[0], VI_game_mdp.dra.get_transition(VI_game_mdp.L[state], state[1])) for
                                     state in VI_game_mdp.states if 'q0' in state])
    ##### SOLVE #####
    VI_game_mdp.solve(do_print=do_print, method='valueIteration', write_video=False,
                      policy_keys_to_print=policy_keys_to_print)
    EM_game_mdp.solve(do_print=do_print, method='expectationMaximization', write_video=False,
                      policy_keys_to_print=policy_keys_to_print)

    policy_difference = MDP.comparePolicies(VI_game_mdp.policy, EM_game_mdp.policy, policy_keys_to_print,
                                            compare_to_decimals=3, do_print=do_print, compute_kl_divergence=True,
                                            reference_policy_has_augmented_states=True,
                                            compare_policy_has_augmented_states=True)
    # Solved mdp.
    return EM_game_mdp, VI_game_mdp, policy_keys_to_print, policy_difference


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
    def __init__(self, maze_cells, cmap):
        # Drop last row and column from maze_cells due to formatting decision for super class.
        super(self.__class__, self).__init__(maze_cells[:-1, :-1], cmap)

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
                bar_height = np.array([phi_at_state[state][act][len(action_list)*(cell)+action_list.index(act)] for state
                    in range(grid_map.size)]).reshape(grid_dim)
            except:
                # Determine type of error to raise when cell is invalid.
                import pdb; pdb.set_trace()
        elif kernels is not None:
            bar_height = np.array([kernels[cell](state) for state in range(grid_map.size)]).reshape(grid_dim)
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

    def confiigurePlot(self, title, policy, action_list, use_print_key, policy_keys_to_print, decimals,
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


# Entry point when called from Command line.
if __name__=='__main__':
    # Program control flags.
    make_new_mdp = False
    write_mdp_policy_csv = True
    gather_new_data = False
    perform_new_inference = True
    inference_method='default' # Default chooses gradient ascent. Other options: 'MLE'
    plot_all_grids = False
    plot_initial_mdp_grids = True
    plot_inferred_mdp_grids = False
    plot_new_phi = False
    plot_new_kernel = False
    plot_loaded_phi = False
    plot_loaded_kernel = False
    plot_flags = [plot_all_grids, plot_initial_mdp_grids, plot_inferred_mdp_grids, plot_new_phi, plot_loaded_phi,
                  plot_new_kernel, plot_loaded_kernel]
    if plot_new_kernel and plot_loaded_kernel:
        raise ValueError('Can not plot both new and loaded kernel in same call.')
    if plot_new_kernel:
        raise NotImplementedError('option: plot_new_kernel doesn\'t work yet. Sorry, but plot_new_phi works!')
    if plot_new_phi and plot_loaded_phi:
        raise ValueError('Can not plot both new and loaded phi in same call.')

    if make_new_mdp:
        EM_mdp, VI_mdp, policy_keys_to_print, policy_difference = makeGridMDPxDRA(do_print=True)
        mdp_file = getOutFile()
        with open(mdp_file, 'w+') as _file:
            print "Pickling EM_mdp to {}".format(mdp_file)
            pickle.dump([EM_mdp, VI_mdp, policy_keys_to_print,policy_difference], _file)
    else:
        # Manually choose file here:
        mdp_file = os.path.join(mdp_obj_path, 'EM_MDP_UTC180117_2135')
        print "Loading file {}.".format(mdp_file)
        with open(mdp_file) as _file:
            EM_mdp, VI_mdp, policy_keys_to_print, policy_difference = pickle.load(_file)
    if write_mdp_policy_csv:
        diff_csv_dict = {key: policy_difference[key] for key in policy_keys_to_print}
        with open(mdp_file+'_Policy_difference.csv', 'w+') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in diff_csv_dict.items():
                for subkey, sub_value in value.items():
                    writer.writerow([key,subkey,sub_value])
        #EM_csv_dict = {key: EM_mdp.policy[key] for key in policy_keys_to_print}
        #with open(mdp_file+'_EM_Policy.csv', 'w+') as csv_file:
        #    writer = csv.writer(csv_file)
        #    for key, value in EM_csv_dict.items():
        #        for subkey, sub_value in value.items():
        #            writer.writerow([key,subkey,sub_value])
        #VI_csv_dict = {key: VI_mdp.policy[key] for key in policy_keys_to_print}
        #with open(mdp_file+'_VI_policy.csv', 'w+') as csv_file:
        #    writer = csv.writer(csv_file)
        #    for key, value in VI_csv_dict.items():
        #        for subkey, sub_value in value.items():
        #            writer.writerow([key,subkey,sub_value])

    # Choose which policy to use for demonstration.
    mdp = EM_mdp

    if gather_new_data:
        # Use policy to simulate and record results.
        #
        # Current policy E{T|R} 6.7. Start by simulating 10 steps each episode.
        num_episodes = 500
        steps_per_episode = 50
        run_histories = np.zeros([num_episodes, steps_per_episode], dtype=np.int8)
        for episode in range(num_episodes):
            # Create time-history for this episode.
            run_histories[episode, 0] = mdp.resetState()
            for t_step in range(1, steps_per_episode):
                run_histories[episode, t_step] = mdp.step()
        # Save sampled trajectories.
        history_file = getOutFile(os.path.basename(mdp_file)
                                  + ('_HIST_{}eps{}steps'.format(num_episodes, steps_per_episode)), data_path)
        with open(history_file, 'w+') as _file:
            print "Pickling Episode histories to {}.".format(history_file)
            pickle.dump(run_histories, _file)
    else:
        # Manually choose data to load here:
        history_file = os.path.join(data_path, 'EM_MDP_UTC180117_2135_HIST_500eps50steps_UTC180117_2154')
        print "Loading history data file {}.".format(history_file)
        with open(history_file) as _file:
            run_histories = pickle.load(_file)
        num_episodes = run_histories.shape[0]
        steps_per_episode = run_histories.shape[1]

    # Determine which states are goals or obstacles.
    normal_states = {state: True if label==empty else False for state, label in labels.items()}
    unique, starting_counts = np.unique(run_histories[:,0], return_counts=True)
    num_trials_from_state = {int(state):0 for state in states}
    num_trials_from_state.update(dict(zip(unique, starting_counts)))
    num_rewards_from_state = {int(state):0 for state in states}
    for run_idx in range(num_episodes):
        starting_state = run_histories[run_idx][0]
        final_state = run_histories[run_idx][-1]
        if final_state==goal_state:
            num_rewards_from_state[starting_state] += 1
    print("In this demonstration 'history' there are  {} episodes, each with {} moves.".format(num_episodes,
          steps_per_episode))
    for state in range(len(states)):
        reward_likelihood = float(num_rewards_from_state[state]) / float(num_trials_from_state[state]) if \
            num_trials_from_state[state] > 0 else np.nan
        print("State {}: Num starts = {}, Num Rewards = {}, likelihood = {}.".format(state,
                                                                                     num_trials_from_state[state],
                                                                                     num_rewards_from_state[state],
                                                                                     reward_likelihood))

    if plot_new_phi or  plot_new_kernel or perform_new_inference:
        tic = time.clock()
        # Solve for approximated observed policy.
        # Use a new mdp to model created/loaded one and a @ref GridGraph object to record, and seach for shortest paths
        # between two grid-cells.
        infer_mdp = MDP(init=initial_state, action_list=action_list, states=states, act_prob=deepcopy(act_prob),
                        grid_map=grid_map)
        infer_mdp.init_set = infer_mdp.states
        graph = GridGraph(grid_map=grid_map, neighbor_dict=infer_mdp.neighbor_dict, label_dict=labels)
        infer_mdp.graph = graph
        # Geodesic Gaussian Kernels, defined as Eq. 3.2 in Statistical Reinforcement
        # Learning, Sugiyama, 2015.
        infer_mdp.ggk_sig = 5.0
        infer_mdp.kernel_centers = [0, 7, 120, 127, 18, 21, 106, 109, 60]
        print ' Performing inference with kernels at:'
        pprint(infer_mdp.kernel_centers)
        # Note that this needs to be the same instance of `GridGraph` assigned to the MDP!
        infer_mdp.gg_kernel_func = lambda s_i, C_i: np.exp(-(float(infer_mdp.graph.shortestPathLength(s_i, C_i)))**2/
                                                           (2*float(infer_mdp.ggk_sig)**2))
        # Note that we need to use a keyword style argument passing to ensure that
        # each lambda function gets its own value of C.
        K = [lambda s, C=cent: infer_mdp.gg_kernel_func(s, C)
             for cent in infer_mdp.kernel_centers]
        # It could be worth pre-computing all of the feature vectors for a small
        # grid...
        infer_mdp.addKernels(K)
        infer_mdp.precomputePhiAtState()
        if not perform_new_inference and (plot_new_phi or plot_new_kernel):
            # Deepcopy the infer_mdp to another variable because and old inference will be loaded into `infer_mdp`.
            new_infer_mdp = deepcopy(infer_mdp)
    if perform_new_inference:
        # Infer the policy from the recorded data.

        if inference_method == 'MLE':
            infer_mdp.inferPolicy(method='historyMLE', histories=run_histories, do_print=True)
        else:
            infer_mdp.inferPolicy(histories=run_histories, do_print=True, use_precomputed_phi=True)
        toc = time.clock() -tic
        print 'Total time to infer policy: {} sec, or {} min.'.format(toc, toc/60.0)
        infered_mdp_file = getOutFile(os.path.basename(history_file) + '_Policy', infered_mdps_path)
        with open(infered_mdp_file, 'w+') as _file:
            print "Pickling Infered Policy to {}.".format(infered_mdp_file)
            pickle.dump(infer_mdp, _file)
    else:
        # Manually choose data to load here:
        infered_mdp_file = os.path.join(infered_mdps_path,
                'EM_MDP_UTC180112_1643_HIST_100eps20steps_UTC180112_1646_Policy_UTC180113_1915')
        print "Loading infered policy data file {}.".format(infered_mdp_file)
        with open(infered_mdp_file) as _file:
            infer_mdp = pickle.load(_file)        # Reconsturct Policy with Q(s,a) = <theta, phi(s,a)>
    if write_mdp_policy_csv:
        infer_csv_dict = {key: infer_mdp.policy[key] for key in infer_mdp.policy.keys()}
        with open(infered_mdp_file+'.csv', 'w+') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in infer_csv_dict.items():
                for subkey, sub_value in value.items():
                    writer.writerow([key,subkey,sub_value])

    # Remember that variable @ref mdp is used for demonstration.
    if len(policy_keys_to_print) == infer_mdp.num_states:
        infered_policy_difference = MDP.comparePolicies(mdp.policy, infer_mdp.policy, policy_keys_to_print,
                                                        compare_to_decimals=3, do_print=True,
                                                        compare_policy_has_extra_keys=False,
                                                        compute_kl_divergence=True)

        kl_divergence_from_demonstration = infer_mdp.computeKLDivergenceOfPolicyFromHistories(run_histories)
        print('The KL-Divergence from the demonstration set is: {0:.3f}.'.format(kl_divergence_from_demonstration))
    else:
        warnings.warn('Demonstration MDP and inferred MDP do not have the same number of states. Perhaps one was '
                      'loaded from an old file? Not printing policy difference.')

    if any(plot_flags):
        # Create plots for comparison. Note that the the `maze` array has one more row and column than the `grid` for
        # plotting purposes.
        maze = np.zeros(np.array(grid_dim)+1)
        for state, label in labels.iteritems():
            if label==red:
                grid_row, grid_col = np.where(grid_map==int(state))
                maze[grid_row, grid_col] = 2
            if label==green:
                grid_row, grid_col = np.where(grid_map==int(state))
                maze[grid_row, grid_col] = 1
        cmap = mcolors.ListedColormap(['w','green','red'])

    plot_policies = []
    only_use_print_keys = []
    titles = []
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
        base_policy_grid = PlotPolicy(maze, cmap, center_offset)
        for policy, use_print_keys, title, kernel_loc in zip(plot_policies, only_use_print_keys, titles,
                                                             kernel_locations):
            # Reorder policy dict for plotting.
            if use_print_keys: # VI and EM policies have DRA states in policy keys.
                list_of_tuples = [(key, policy[key]) for key in policy_keys_to_print]
            else: # Learned policy only has state numbers.
                order_of_keys = [str(key) for key in range(grid_map.size)]
                list_of_tuples = [(key, policy[key]) for key in order_of_keys]
            policy = OrderedDict(list_of_tuples)
            fig = base_policy_grid.confiigurePlot(title, policy, action_list, use_print_keys, policy_keys_to_print,
                                                  decimals=2, kernel_locations=kernel_loc)

        print '\n\nHEY! You! With the face! (computers don\'t have faces) Mazimize figure window to correctly show ' \
                'arrow/dot size ratio!\n'

    if plot_loaded_kernel or plot_new_kernel:
        if not perform_new_inference and plot_new_kernel:
            kernels = new_infer_mdp.kernels
        else:
            kernels = infer_mdp.kernels
        kernel_grid =PlotKernel(maze, cmap)
        kern_idx = 0
        title='Kernel Centered at {}.'.format(kern_idx)
        fig, ax =  kernel_grid.configurePlot(title, kern_idx, kernels=kernels)

    if plot_loaded_phi or plot_new_phi:
        if not perform_new_inference:
            phi_at_state = new_infer_mdp.phi_at_state
        else:
            phi_at_state = infer_mdp.phi_at_state
        phi_grid =PlotKernel(maze, cmap)
        phi_idx = 0
        act = 'Empty'
        title='Phi Values Centered at {} for action {}.'.format(phi_idx, act)
        fig, ax =  phi_grid.configurePlot(title, phi_idx, phi_at_state=phi_at_state, act=act)

    if any(plot_flags):
        plt.show()

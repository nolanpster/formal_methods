#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu, Nishant Doshi, ndoshi@wpi.edu'


from kernel_centers import GeodesicGaussianKernelCenter as GGKCent
from kernel_centers import OrdinaryGaussianKernelCenter as OGKCent

from copy import copy
import numpy as np


class FeatureVector(object):
    """
    @brief @todo
    """
    np_all = np.s_[:] # Slice operator for all values.

    def __init__(self, action_list, trans_prob_function, graph, ggk_centers=frozenset([]), ogk_centers=frozenset([]),
                 std_devs=None, dtype=np.float64, ggk_mobile_indices=[], ogk_mobile_indices=[], state_list=[],
                 state_idx_to_infer=0, mobile_kernel_state_idx=None):
        """
        @brief Creates and instance of a feature vector class.

        @param action_lsit The list of actions available
        @param trans_prob_function A reference  to a transition probability function that takes in a state, and action
               and returns a vector of the transition probabilities to the next states. See @ref MDP.T().
        @param graph A reference to a GridGraph object.
        @param ggk_centers a list of length G of Geodesic Gaussian Kernel locations.
        @param ogk_centers a list of length O of Ordinary Gaussian Kernel locations.
        @param std_devs a @ref numpy.array() of length G+O of variances, default is 1.
        @param dtype Data type to use for calculations, default is numpy.float64.
        @param ggk_mobile_indices List of indices of any GGK center that are a reference to the current grid cell of any
               mobile agent. Note that these should be relative to the @ref ggk_centers argument.
        @param ogk_mobile_indices List of indices of any OGK center that are a reference to the current grid cell of any
               mobile agent. Note that these should be relative to the @ref ogk_centers argument.
        @param state_list a list of tuples of states that list the grid cell locations of any agents.

        @note mobile kernels are only implemented for GGKs and they're restricted to being at the end of the list
        of ggk_centers (e.g, after all centers of fixe kernels). (3/13/18)

        @note Unless otherwise noted, all axes are referenced as:
                - k : kernel axis
                - l : state-axis
                - m : action-axis
                - n : phi axis (length |j|*|k|)
        """
        # Required argument semantics.
        self.action_list = action_list
        self.trans_prob_func = trans_prob_function
        self.graph = graph
        self.dtype = dtype
        self.state_idx_to_infer = state_idx_to_infer
        self.mobile_kernel_state_idx = mobile_kernel_state_idx

        # Prepare Geodesic Gaussian Kernels
        self.ggk_centers = ggk_centers
        self.G = len(self.ggk_centers)
        self.GGK_dist_objs = [GGKCent(cent, self.graph) for cent in self.ggk_centers]

        # Prepare Ordinary Gaussian Kernels
        self.ogk_centers = ogk_centers
        self.O = len(self.ogk_centers)
        self.OGK_dist_objs = [OGKCent(cent, self.graph) for cent in self.ogk_centers]

        # Objects that return the distance to each kernel center when called.
        self.kernel_dist_objs = self.GGK_dist_objs + self.OGK_dist_objs

        # Configure general properties of the FeatureVector function.
        self.states = state_list
        self.num_states = len(self.states)
        self.state_indices = range(self.num_states)
        self.grid_cell_vec = self.graph.grid_map.ravel()
        self.num_cells = len(self.grid_cell_vec)
        self.num_actions = len(action_list)
        self.num_kernels = self.G + self.O
        self.kernel_centers = list(self.ggk_centers) + list(self.ogk_centers)
        self.length = self.num_kernels * self.num_actions

        # Create a list of any mobile kernels.
        shifted_ogk_mobile_indices = [self.G + ogk_mob_idx for ogk_mob_idx in ogk_mobile_indices]
        self.mobile_indices = ggk_mobile_indices + shifted_ogk_mobile_indices
        self.num_mobile_kernels = len(self.mobile_indices)
        self.has_mobile_kernels = True if self.num_mobile_kernels > 0 else False

        if self.has_mobile_kernels:
            self.num_static_kernels = self.num_kernels - self.num_mobile_kernels
            self.num_calc_array_rows = self.num_static_kernels + ( self.num_mobile_kernels * self.num_cells)
        else:
            self.num_static_kernels = self.num_kernels
            self.num_calc_array_rows = self.num_static_kernels
        self.static_kernel_indices = range(self.num_static_kernels)
        # Replicate entries for any "mobile" kernels for each state.
        self.expandKernDistObjs()
        self.calcMobileIndicesForState()

        if std_devs is not None:
            self.std_devs = std_devs
            if len(std_devs) != self.num_kernels:
                raise ValueError('Length of Provided Standard deviation vector, {} not equal to the number of kernels, '
                                 '{}!'.format(len(std_devs), self.num_kernels))
        else:
            self.std_devs = np.ones(self.num_kernels)
        self.expandStdDevVec()

        # Build initial set of matrices
        self.cell_distances_to_kernels = np.empty([self.num_calc_array_rows, self.num_cells], dtype=self.dtype)
        self.kernel_argument = np.empty([self.num_calc_array_rows, self.num_cells], dtype=self.dtype)
        self.kernel_values = np.empty([self.num_calc_array_rows, self.num_cells], dtype=self.dtype)
        self.weighted_prob_kernel_sum = np.empty([self.num_calc_array_rows, self.num_cells, self.num_actions])
        self.buildTransProbMat()
        self.updateCellDistancesToKernels()
        self.updateStdDevs(also_update_kernel_weights=True)

        # Variables to keep track of things that have already been calculated.
        self.evaluatedStateActions = frozenset([])

    def __len__(self):
        """
        @brief returns the length of the feature vector
        """
        return self.length

    def __call__(self, state, action):
        """
        @brief Evaluate the vector of basis functions, phi. All kernels are multiplied an action indicator function. A
               feature vector will have @c m*p elements, where @c m is the number of actions, and @c p is the number of
               kernels.

        @param state A tuple representing a state that exists in FeatureVector.states.
        @param action An action in the action list.
        """
        if self.has_mobile_kernels:
            indices_to_return = self.mobilePhiIndices[state]
        else:
            indices_to_return = self.static_kernel_indices

        action_idx = self.action_list.index(action)
        phi_mat = np.zeros([self.num_actions, self.num_kernels])
        phi_mat[action_idx, :] = self.weighted_prob_kernel_sum[indices_to_return, state[self.state_idx_to_infer],
                                                               action_idx]

        return phi_mat.transpose().ravel().transpose()

    def calcMobileIndicesForState(self):
        self.mobilePhiIndices = {state: self.static_kernel_indices
                                        + [self.num_static_kernels + (mob_idx * state[self.mobile_kernel_state_idx])
                                           for mob_idx in range(1, self.num_mobile_kernels+1)] for state in self.states}

    def expandStdDevVec(self):
        self.expanded_std_devs = np.empty([self.num_calc_array_rows])
        self.expanded_std_devs[:self.num_static_kernels] = self.std_devs[:self.num_static_kernels]

        num_static_k = self.num_static_kernels
        for mob_k_idx in self.mobile_indices:
            self.expanded_std_devs[num_static_k: ((mob_k_idx+1) * self.num_cells) + num_static_k] = self.std_devs[mob_k_idx]

    def expandKernDistObjs(self):
        num_static_k = self.num_static_kernels
        self.expanded_kernel_dist_objs = copy(self.kernel_dist_objs[:num_static_k])
        for mob_k_idx in self.mobile_indices:
            kern_at_every_state = [GGKCent(cell, self.graph) for cell in self.grid_cell_vec]
            self.expanded_kernel_dist_objs += kern_at_every_state

    def buildTransProbMat(self):
        self.prob_mat = np.empty([self.num_cells, self.num_cells, self.num_actions], dtype=self.dtype)
        for cell in self.grid_cell_vec:
            for act_idx, action in enumerate(self.action_list):
                self.prob_mat[cell, :, act_idx] = self.trans_prob_func(cell, action)

    def updateStdDevs(self, std_devs=None, also_update_kernel_weights=True):
        if std_devs is not None:
            self.std_devs = std_devs
            self.expandStdDevs()
        self.kernel_divisor_reciprocal = np.reciprocal(2*np.power(self.expanded_std_devs, 2, dtype=self.dtype))

        if also_update_kernel_weights:
            self.updateKernels()
            self.updateWeightedSum()

    def updateKernelWeights(self, selected_kernel_indices=None):
        """
        @brief Updates the evaluations of the kernels.

        @param selected_kernel_indices The indices to update the selected indices, otherwise all will be updated.
        """
        self.updateCellDistancesToKernels(selected_kernel_indices)
        self.updateKernels(selected_kernel_indices)
        self.updateWeightedSum(selected_kernel_indices)

    def updateCellDistancesToKernels(self, selected_kernel_indices=None):
        """
        @brief Updates the distances to kernel centers.

        @param selected_kernel_indices The indices to update the selected indices, otherwise all will be updated.
        """
        if selected_kernel_indices is not None:
            kern_iterator = zip(selected_kernel_indices, self.expanded_kernel_dist_objs[selected_kernel_indices])
        else:
            # Update all kernels
            kern_iterator = enumerate(self.expanded_kernel_dist_objs)

        for kern_idx, kern_dist in kern_iterator:
            # Running kern_dist(state_num) will return the distance by the metric of the specific kernel.
            self.cell_distances_to_kernels[kern_idx] = map(kern_dist, self.grid_cell_vec)

    def updateKernels(self, selected_kernel_indices=None):
        """
        @brief Updates the values of kernels.

        @param selected_kernel_indices The indices to update the selected indices, otherwise all will be updated.
        """
        if selected_kernel_indices is None:
            # Slice with ':' instead of 'None' because 'None' adds an extra dimension to the array.
            selected_kernel_indices = self.np_all

        kernel_arg_numerator = np.negative(np.power(self.cell_distances_to_kernels[selected_kernel_indices], 2))
        kernel_arg_denom_recip = self.kernel_divisor_reciprocal[selected_kernel_indices]
        self.kernel_argument[selected_kernel_indices] = np.einsum('kl,k->kl', kernel_arg_numerator,
                                                                  kernel_arg_denom_recip)
        self.kernel_values[selected_kernel_indices] = np.exp(self.kernel_argument[selected_kernel_indices])

    def updateWeightedSum(self, selected_kernel_indices=None):
        """
        @brief Updates the weighted sum of P(s'|s,a)*K(s',c) for all kernsl, states and actions.

        @param selected_kernel_indices The indices to update the selected indices, otherwise all will be updated.
        """
        if selected_kernel_indices is None:
            # Slice with ':' instead of 'None' because 'None' adds an extra dimension to the array.
            selected_kernel_indices = xrange(self.num_calc_array_rows)

        # Below, the prob-mat has dimsion |S|x|S|x|A|, the first axis of states is indexed with 'i'.
        for kernel_idx in selected_kernel_indices:
            for act_idx in xrange(self.num_actions):
                for cell in self.grid_cell_vec:
                    self.weighted_prob_kernel_sum[kernel_idx, cell, act_idx] = \
                        np.dot(self.kernel_values[kernel_idx], self.prob_mat[cell, :, act_idx])

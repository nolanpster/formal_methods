#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu, Nishant Doshi, ndoshi@wpi.edu'


from kernel_centers import GeodesicGaussianKernelCenter as GGKCent
from kernel_centers import OrdinaryGaussianKernelCenter as OGKCent
import numpy as np


class FeatureVector(object):
    """
    @brief @todo
    """
    np_all = np.s_[:] # Slice operator for all values.

    def __init__(self, action_list, trans_prob_function, graph, ggk_centers=None, ogk_centers=None, std_devs=None,
                 dtype=np.float64, ggk_mobile_indeces=[], ogk_mobile_indeces=[]):
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
        @param ggk_mobile_indeces List of indeces of any GGK center that are a reference to the current grid cell of any
               mobile agent. Note that these should be relative to the @ref ggk_centers argument.
        @param ogk_mobile_indeces List of indeces of any OGK center that are a reference to the current grid cell of any
               mobile agent. Note that these should be relative to the @ref ogk_centers argument.

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

        # Prepare Geodesic Gaussian Kernels
        self.ggk_centers = ggk_centers
        self.G = len(self.ggk_centers) if self.ggk_centers is not None else 0
        self.GGK_dist_objs = [GGKCent(cent, graph) for cent in self.ggk_centers]

        # Prepare Ordinary Gaussian Kernels
        self.ogk_centers = ogk_centers
        self.O = len(self.ogk_centers) if self.ogk_centers is not None else 0
        self.OGK_dist_objs = [OGKCent(cent, graph) for cent in self.ogk_centers]

        # Objects that return the distance to each kernel center when called.
        self.kernel_dist_objs = self.GGK_dist_objs + self.OGK_dist_objs

        # Configure general properties of the FeatureVector function.
        self.state_vec = self.graph.grid_map.ravel()
        self.num_states = len(self.state_vec)
        self.num_actions = len(action_list)
        self.num_kernels = self.G + self.O
        self.length = self.num_kernels * self.num_actions

        # Create a list of any mobile kernels.
        shifted_ogk_mobile_indeces = [self.G + ogk_mob_ind for ogk_mob_ind in ogk_mobile_indeces]
        self.mobile_indeces = ggk_mobile_indeces + shifted_ogk_mobile_indeces
        self.has_mobile_kernels = True if len(self.mobile_indeces) > 0 else False

        if std_devs is not None:
            self.std_devs = std_devs
            if len(std_devs) != self.num_kernels:
                raise ValueError('Length of Provided Standard deviation vector, {} not equal to the number of kernels, '
                                 '{}!'.format(len(std_devs), self.num_kernels))
        else:
            self.std_devs = 1

        # Build initial set of matrices
        self.state_distances_to_kernels = np.empty([self.num_kernels, self.num_states], dtype=self.dtype)
        self.kernel_argument = np.empty([self.num_kernels, self.num_states], dtype=self.dtype)
        self.kernel_values = np.empty([self.num_kernels, self.num_states], dtype=self.dtype)
        self.dkernel_values = np.empty([self.num_kernels, self.num_states], dtype=self.dtype)
        self.weighted_prob_kernel_sum = np.empty([self.num_kernels, self.num_states, self.num_actions])
        self.buildTransProbMat()
        self.updateStateDistancesToKernels()
        self.updateStdDevs(also_update_kernel_weights=True)
        self.buildKernelDeltas()

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
               kernels. This function takes arguments (<str>state, <str>action).
        """
        if self.has_mobile_kernels:
            self.updateKernelWeights(self.mobile_indeces)

        this_action_idx = self.action_list.index(action)
        phi_mat = np.zeros([self.num_actions, self.num_kernels])
        phi_mat[this_action_idx, :] = self.weighted_prob_kernel_sum[:, state, this_action_idx]

        return phi_mat.transpose().ravel().transpose()

    def buildTransProbMat(self):
        self.prob_mat = np.empty([self.num_states, self.num_states, self.num_actions], dtype=self.dtype)
        for state in self.state_vec:
            for act_idx, action in enumerate(self.action_list):
                self.prob_mat[state, :, act_idx] = self.trans_prob_func(state, action)

    def updateStdDevs(self, std_devs=None, also_update_kernel_weights=True):
        if std_devs is not None:
            self.std_devs = std_devs
        self.kernel_divisor_reciprocal = np.reciprocal(2*np.power(self.std_devs, 2, dtype=self.dtype))
        self.dkernel_divisor_reciprocal = np.reciprocal(np.power(self.std_devs, 3, dtype=self.dtype))

        if also_update_kernel_weights:
            self.updateKernels()
            self.updateWeightedSum()

    def updateKernelWeights(self, selected_kernel_indeces=None):
        """
        @brief Updates the evaluations of the kernels.

        @param selected_kernel_indeces The indeces to update the selected indeces, otherwise all will be updated.
        """
        self.updateStateDistancesToKernels(selected_kernel_indeces)
        self.updateKernels(selected_kernel_indeces)
        self.updateWeightedSum(selected_kernel_indeces)

    def updateStateDistancesToKernels(self, selected_kernel_indeces=None):
        """
        @brief Updates the distances to kernel centers.

        @param selected_kernel_indeces The indeces to update the selected indeces, otherwise all will be updated.
        """
        if selected_kernel_indeces is not None:
            kern_iterator = zip(selected_kernel_indeces, self.kernel_dist_objs[selected_kernel_indeces])
        else:
            # Update all kernels
            kern_iterator = enumerate(self.kernel_dist_objs)

        for kern_idx, kern_dist in kern_iterator:
            # Running kern_dist(state_num) will return the distance by the metric of the specific kernel.
            self.state_distances_to_kernels[kern_idx] = map(kern_dist, self.state_vec)

    def updateKernels(self, selected_kernel_indeces=None):
        """
        @brief Updates the values of kernels.

        @param selected_kernel_indeces The indeces to update the selected indeces, otherwise all will be updated.
        """
        if selected_kernel_indeces is None:
            # Slice with ':' instead of 'None' because 'None' adds an extra dimension to the array.
            selected_kernel_indeces = self.np_all

        kernel_arg_numerator = np.negative(np.power(self.state_distances_to_kernels[selected_kernel_indeces], 2))
        kernel_arg_denom_recip = self.kernel_divisor_reciprocal[selected_kernel_indeces]
        self.kernel_argument[selected_kernel_indeces] = np.einsum('kl,k->kl', kernel_arg_numerator,
                                                                  kernel_arg_denom_recip)
        self.kernel_values[selected_kernel_indeces] = np.exp(self.kernel_argument[selected_kernel_indeces])

    def updateWeightedSum(self, selected_kernel_indeces=None):
        """
        @brief Updates the weighted sum of P(s'|s,a)*K(s',c) for all kernsl, states and actions.

        @param selected_kernel_indeces The indeces to update the selected indeces, otherwise all will be updated.
        """
        if selected_kernel_indeces is None:
            # Slice with ':' instead of 'None' because 'None' adds an extra dimension to the array.
            selected_kernel_indeces = xrange(self.num_kernels)

        # Below, the prob-mat has dimsion |S|x|S|x|A|, the first axis of states is indexed with 'i'.
        for kernel_ind in selected_kernel_indeces:
            for act_ind in xrange(self.num_actions):
                for state in self.state_vec:
                    self.weighted_prob_kernel_sum[kernel_ind, state, act_ind] = \
                        np.inner(self.kernel_values[kernel_ind], self.prob_mat[state, :, act_ind])


    def buildKernelDeltas(self, selected_kernel_indeces=None):
            """
            @brief Updates the weighted sum of P(s'|s,a)*K(s',c) for all kernsl, states and actions.

            @param selected_kernel_indeces The indeces to update the selected indeces, otherwise all will be updated.
            """
            if selected_kernel_indeces is None:
                # Slice with ':' instead of 'None' because 'None' adds an extra dimension to the array.
                selected_kernel_indeces = xrange(self.num_kernels)
            kernel_arg_numerator = np.power(self.state_distances_to_kernels[selected_kernel_indeces], 2)
            dkernel_arg_denom_recip = self.dkernel_divisor_reciprocal[selected_kernel_indeces]
            self.dkernel_values[selected_kernel_indeces] = np.einsum('kl,k->kl', kernel_arg_numerator,
                                                                  dkernel_arg_denom_recip)

            # Below, the prob-mat has dimsion |S|x|S|x|A|, the first axis of states is indexed with 'i'.
            for kernel_ind in selected_kernel_indeces:
                for act_ind in xrange(self.num_actions):
                    for state in self.state_vec:
                        self.weighted_prob_kernel_sum[kernel_ind, state, act_ind] = \
                            np.inner(self.kernel_values[kernel_ind]*self.dkernel_values[kernel_ind], self.prob_mat[state, :, act_ind])
            return self.weighted_prob_kernel_sum

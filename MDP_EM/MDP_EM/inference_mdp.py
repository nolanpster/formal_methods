#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu, Nishant Doshi, ndoshi@wpi.edu'
from policy_inference import PolicyInference
from grid_graph import GridGraph
from feature_vector import FeatureVector
from copy import deepcopy
from MDP import MDP
import numpy as np


class InferenceMDP(MDP):
    """A Markov Decision Process, defined by an initial state, transition model --- the probability transition matrix,
    np.array prob[a][0,1] -- the probability of going from 0 to 1 with action a.  and reward function. We also keep
    track of a gamma value, for use by algorithms. The transition model is represented somewhat differently from the
    text.  Instead of T(s, a, s') being probability number for each state/action/state triplet, we instead have T(s, a)
    return a list of (p, s') pairs.  We also keep track of the possible states, terminal states, and actions for each
    state.  The input transitions is a dictionary: (state,action): list of next state and probability tuple.  AP: a set
    of atomic propositions. Each proposition is identified by an index between 0 -N.  L: the labeling function,
    implemented as a dictionary: state: a subset of AP."""
    def __init__(self, init=None, action_list=[], states=[], prob=dict([]), gamma=.9, AP=set([]), L=dict([]),
                 reward=dict([]), grid_map=None, act_prob=dict([]), gg_kernel_centers=frozenset([]),
                 og_kernel_centers=frozenset([]), kernel_sigmas=None, prob_dtype=np.float64, state_idx_to_observe=0,
                 fixed_obstacle_labels=dict([]), ggk_mobile_indices=[], ogk_mobile_indices=[],
                 state_idx_of_mobile_kernel=None):
        """
        @brief Construct an MDP meant to perform inference.
        @param init @todo
        @param action_list @todo
        @param states @todo
        @param prob A list of tuples of states.
        @param acc @todo
        @param gamma @todo
        @param AP @todo
        @param L @todo
        @param reward @todo
        @param grid_map @todo
        @param act_prob @todo
        @param ggk_centers a list of length G of Geodesic Gaussian Kernel locations in the grid.
        @param ogk_centers a list of length O of Ordinary Gaussian Kernel locations in the grid.
        @param kernel_sigmas a @ref numpy.array() of length G+O of standard deviations.
        @param prob_dtype Data type to use for storing transition probabilities.
        @param state_idx_to_observe The indices of a state tuple that are "observations". Primarily used by
               GridGraph.getOvservedAction().
        """
        super(self.__class__, self).__init__(init=init, action_list=action_list, states=states, prob=prob, gamma=gamma,
                                             AP=AP, L=L, reward=reward, grid_map=grid_map, act_prob=act_prob,
                                             prob_dtype=prob_dtype)
        if not fixed_obstacle_labels:
            self.fixed_obstacle_labels = self.L
        else:
            self.fixed_obstacle_labels = fixed_obstacle_labels

        self.state_idx_to_observe = state_idx_to_observe
        if self.neighbor_dict is None:
            self.buildNeighborDict()
        if (grid_map is not None) and (self.fixed_obstacle_labels):
            self.graph = GridGraph(grid_map=grid_map, neighbor_dict=self.neighbor_dict,
                                   label_dict=self.fixed_obstacle_labels,
                                   state_idx_to_observe=self.state_idx_to_observe)
        else:
            self.graph = None

        self.gg_kernel_centers = gg_kernel_centers
        self.ggk_mobile_indices = ggk_mobile_indices
        self.og_kernel_centers = og_kernel_centers
        self.ogk_mobile_indices = ogk_mobile_indices
        self.kernel_centers = list(self.gg_kernel_centers) + list(self.og_kernel_centers)
        self.kernel_sigmas = kernel_sigmas
        self.state_idx_of_mobile_kernel = state_idx_of_mobile_kernel
        if (self.graph is not None) and (self.kernel_centers):
            # Graph is not none and kernel centers is not an empty list:
            self.buildKernels()
        else:
            self.phi = None
            self.num_kern = None
            self.phi_at_state = None
        # Theta is used as the policy parameter for inference.
        self.theta = None

        # cell_state_slicer: used to extract indeces from the tuple of states. The joint-state tuples are used as
        # dictionary keys and this class wants to slice all states.
        self.state_slice_length = None
        self.cell_state_slicer = slice(None, self.state_slice_length, None)

    def T(self, state, action):
        """
        Transition model.  From a state and an action, return a row in the
        matrix for next-state probability.
        """
        return self.prob[action][state, :]

    def buildKernels(self, gg_kernel_centers=None, og_kernel_centers=None, kernel_sigmas=None):
        """
        @brief @todo

        @param kernel_centers A set/list/numpy.array of kernel centers to use. If not provided this method assumes that
               the member is already set.
        """
        kernel_centers_were_updated = False
        if kernel_sigmas is not None:
            self.kernel_sigmas = kernel_sigmas
        if gg_kernel_centers is not None:
            self.gg_kernel_centers = gg_kernel_centers
            kernel_centers_were_updated = True
        if og_kernel_centers is not None:
            self.og_kernel_centers = og_kernel_centers
            kernel_centers_were_updated = True
        if kernel_centers_were_updated:
            self.kernel_centers = list(self.gg_kernel_centers) + list(self.og_kernel_centers)

        self.phi = FeatureVector(self.action_list, self.T, self.graph, ggk_centers=self.gg_kernel_centers,
                                 ogk_centers=self.og_kernel_centers, std_devs=self.kernel_sigmas,
                                 state_list=self.states, state_idx_to_infer=self.state_idx_to_observe,
                                 ggk_mobile_indices=self.ggk_mobile_indices,
                                 ogk_mobile_indices=self.ogk_mobile_indices,
                                 mobile_kernel_state_idx=self.state_idx_of_mobile_kernel)
        self.num_kern = self.phi.num_kernels
        self.precomputePhiAtState()

    def precomputePhiAtState(self):
        self.phi_at_state = {state: {act: self.phi(state, act) for act in self.action_list} for state in
                             self.observable_states}

    def updateSigmas(self, sigmas):
        self.phi.updateStdDevs(sigmas)
        self.precomputePhiAtState()

    def buildGibbsPolicy(self):
        """
        @brief Method to build the policy during/after policy inference.
        """
        exp_Q = {state: {act: np.exp(np.dot(self.theta, self.phi_at_state[state][act])) for act in self.action_list}
                 for state in self.observable_states}
        sum_exp_Q = {state: sum(exp_Q[state].values()) for state in self.observable_states}
        self.policy = {state: {act: exp_Q[state][act]/sum_exp_Q[state] for act in self.action_list}
                       for state in self.observable_states}

    def inferPolicy(self, method='gradientAscent', write_video=False, **kwargs):
        """
        @brief Infers the policy of a given MDP. Defaults to the Gradient Ascent method.

        @param an instance of @ref MDP.
        @param a string matching a method name in @ref policy_solvers.py.
        """
        return PolicyInference(self, method=method, write_video=write_video).infer(**kwargs)

    @staticmethod
    def evalGibbsPolicy(Q_dict):
        """
        @brief Evaluates the policy at a single state given Q values.

        @param a dictionary with action keys and  Q-values for values
        @return a policy distribution dictionary
        """
        exp_Q = {act: np.exp(Q_dict[act]) for act in Q_dict.keys()}
        sum_exp_Q = sum(exp_Q.values())
        policy = {act: exp_Q[act]/sum_exp_Q for act in Q_dict.keys()}
        return policy

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
    def __init__(self, init=None, action_list=[], states=[], prob=dict([]), acc=None, gamma=.9, AP=set([]), L=dict([]),
                 reward=dict([]), grid_map=None, act_prob=dict([]), gg_kernel_centers=[], og_kernel_centers=[],
                 kernel_sigmas=None):
        """
        @brief Construct an MDP meant to perform inference.
        @param init @todo
        @param action_list @todo
        @param states @todo
        @param prob @todo
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
        """
        super(self.__class__, self).__init__(init=init, action_list=action_list, states=states, prob=prob, acc=acc,
                                             gamma=gamma, AP=AP, L=L, reward=reward, grid_map=grid_map,
                                             act_prob=act_prob)

        self.graph = GridGraph(grid_map=grid_map, neighbor_dict=self.neighbor_dict, label_dict=self.L)

        self.gg_kernel_centers = gg_kernel_centers
        self.og_kernel_centers = og_kernel_centers
        self.kernel_centers = list(self.gg_kernel_centers) + list(self.og_kernel_centers)
        self.kernel_sigmas = kernel_sigmas
        self.buildKernels()

        # option to rebuild kernels here?

    def buildKernels(self, kernel_centers=None, kernel_sigmas=None):
        """
        @brief @todo

        @param kernel_centers A set/list/numpy.array of kernel centers to use. If not provided this method assumes that
               the member is already set.
        """
        self.phi = FeatureVector(self.action_list, self.T, self.graph, ggk_centers=self.gg_kernel_centers,
                                 ogk_centers=self.og_kernel_centers, std_devs=self.kernel_sigmas)
        self.num_kern = self.phi.num_kernels
        self.precomputePhiAtState()

    def precomputePhiAtState(self):
        self.phi_at_state = {state: {act: self.phi(str(state), act) for act in self.action_list} for state in
                             self.state_vec}

    def updateSigmas(self, sigmas):
        self.phi.updateStdDevs(sigmas)
        self.precomputePhiAtState()

    def inferPolicy(self, method='gradientAscent', write_video=False, **kwargs):
        """
        @brief Infers the policy of a given MDP. Defaults to the Gradient Ascent method.

        @param an instance of @ref MDP.
        @param a string matching a method name in @ref policy_solvers.py.
        """
        return PolicyInference(self, method=method, write_video=write_video).infer(**kwargs)

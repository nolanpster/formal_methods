#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu, Nishant Doshi, ndoshi@wpi.edu'
from policy_inference import PolicyInference
from grid_graph import GridGraph
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
                 reward=dict([]), grid_map=None, act_prob=dict([]), kernel_type=None, kernel_sigma=1.0,
                 kernel_centers=None):
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
        @param kernel_type Kernel type to use for inference, select from ['GGK',], default is 'GGK' that ues Geodesic
               Gaussian Kernels defined as Eq. 3.2 in Statistical Reinforcement Learning, Sugiyama, 2015.
        @param kernel_sigma Standard deviation of Kernels.
        @param kernel_centers Grid cell locations of the kernels.
        """
        super(self.__class__, self).__init__(init=init, action_list=action_list, states=states, prob=prob, acc=acc,
                                             gamma=gamma, AP=AP, L=L, reward=reward, grid_map=grid_map,
                                             act_prob=act_prob)

        self.graph = GridGraph(grid_map=grid_map, neighbor_dict=self.neighbor_dict, label_dict=self.L)

        if kernel_type is None:
            self.kernel_type = 'GGK'
        else:
            self.kernel_type = kernel_type
        self.setKernelSigma(kernel_sigma)
        if (kernel_centers is not None) and (self.kernel_type=='GGK'):
            self.buildGGKs(kernel_centers)

    def addKernels(self, kernels):
        self.kernels = kernels
        self.num_kern = len(self.kernels)

    def setKernelSigma(self, kernel_sigma):
        self.kernel_sigma = kernel_sigma

    def setKernelCenters(self, kernel_centers):
        self.kernel_centers = kernel_centers

    def addEuclideanKernels(self,K):
        self.EKerns=K

    def buildGGKs(self, kernel_centers=None, kernel_sigma=None):
        """
        @brief @todo

        @param kernel_centers A set/list/numpy.array of kernel centers to use. If not provided this method assumes that
               the member is already set.
        """
        if kernel_centers is not None:
            self.setKernelCenters(kernel_centers)
        if kernel_sigma is not None:
            sef.setKernelSigma(kernel_sigma)
        self.gg_kernel_func = lambda s_i, C_i: np.exp(-(float(self.graph.shortestPathLength(s_i, C_i)))**2/
                                                      (2*float(self.kernel_sigma)**2))
        K = [lambda s, C=cent: self.gg_kernel_func(s, C) for cent in self.kernel_centers]
        self.addKernels(K)
        self.precomputePhiAtState()

    def phi(self, state, action):
        # Create vector of basis functions, phi. All kernels are multiplied an action indicator function. A feature
        # vector will have @c m*p elements, where @c m is the number of actions, and @c p is the number of kernels. This
        # function takes arguments (<str>state, <str>action).
        phi = np.zeros([self.num_actions*self.num_kern, 1]) # Column vector
        i_state = int(state)
        for _i, act in enumerate(self.action_list):
            this_ind = lambda a_in, a_i=act: self.act_ind(a_in, a_i)
            if this_ind(action):
                trans_probs = self.T(state, act)
                for _j, kern in enumerate(self.kernels):
                    # Eq. 3.3 Sugiyama 2015
                    kern_weights = np.array(map(kern, self.state_vec))
                    phi[_i+(_j)*self.num_actions] = this_ind(action) * np.inner(trans_probs, kern_weights)
        return phi

    def precomputePhiAtState(self):
        self.phi_at_state = {state: {act: self.phi(str(state), act) for act in self.action_list} for state in
                             self.state_vec}

    def Ephi(self, state, action):
        # Create vector of basis functions, phi. All kernels are multiplied an
        # action indicator function. A feature vector will have @c m*p
        # elements, where @c m is the number of actions, and @c p is the number
        # of kernels. This function takes arguments (<str>state, <str>action).
        phi = np.zeros([self.num_actions*self.num_kern, 1]) # Column vector
        i_state = int(state)
        for _i, act in enumerate(self.action_list):
            this_ind = lambda a_in, a_i=act: self.act_ind(a_in, a_i)
            if this_ind(action):
                trans_probs = self.T(state, act)
                for _j, kern in enumerate(self.Ekerns):
                    # Eq. 3.3 Sugiyama 2015
                    kern_weights = np.array(map(kern, self.state_vec))
                    phi[_i+(_j)*self.num_actions] = \
                        this_ind(action) * np.inner(trans_probs, kern_weights)
        return phi

    def inferPolicy(self, method='gradientAscent', write_video=False, **kwargs):
        """
        @brief Infers the policy of a given MDP. Defaults to the Gradient Ascent method.

        @param an instance of @ref MDP.
        @param a string matching a method name in @ref policy_solvers.py.
        """
        return PolicyInference(self, method=method, write_video=write_video).infer(**kwargs)

#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

import numpy as np
from copy import deepcopy
from pprint import pprint

class PolicyInference(object):
    """
    @brief Class that infers for polices using an instance of the MDP class.

    @pre Assumes the MDP instance has an instance variable .graph of type @ref GridGraph.
    """
    def __init__(self, mdp=None, method=None, write_video=False):
        self.mdp = mdp
        self.setMethod(method)


    def infer(self, method=None, **kwargs):
        if method is not None and (not self.method == method):
            self.setMethod(method)
        # Unbound method call requires passing in 'self'.
        self.algorithm(self, **kwargs)

    def setMethod(self, method='gradientAscent'):
            # Calles a method named 'method'. So self.algorithm points to the
            # method in the PolicyInference class.
            self.method = method
            if self.method is not None:
                self.algorithm = getattr(PolicyInference, method)
            else:
                self.algorithm = None

    def delHistDelThetaEst(self, theta, use_precomputed_phi=False):
        """
        The estimated gradient of the distribution of all histories given a policy, with respect to the policy.

        See Sugiyama 2015 7.2.1 (pg. 97).
        """
        # Probability of actions been chosen given theta and state.
        (num_episodes, num_steps) = self.histories.shape
        grad_wrt_theta = np.zeros(self.theta_size)
        # Pre-compute all possible values (for small environments).
        if use_precomputed_phi:
            exp_Q = {state: {act: np.exp(np.dot(theta, self.mdp.phi_at_state[state][act])) for act in
                             self.mdp.action_list} for state in self.mdp.state_vec}
            sum_exp_Q = {state: sum(exp_Q[state].values()) for state in self.mdp.state_vec}
            phi_weighted_exp_Q = {state: {act: self.mdp.phi_at_state[state][act]*exp_Q[state][act] for act in
                                          self.mdp.action_list} for state in self.mdp.state_vec}
            sum_weighted_exp_Q = {state: sum(phi_weighted_exp_Q[state].values()) for state in self.mdp.state_vec}
            del_theta_total_Q = {state: sum_weighted_exp_Q[state]/sum_exp_Q[state] for state in self.mdp.state_vec}
        else:
            raise NotImplementedError
        for episode in range(num_episodes):
            for t_step in range(1, num_steps):
                this_state = self.histories[episode, t_step-1]
                next_state = self.histories[episode, t_step]
                observed_action = self.mdp.graph.getObservedAction(this_state, next_state)
                if observed_action is None:
                    raise ValueError('No valid action from {} to {}'.format(this_state, next_state))
                del_beta_del_theta = (self.mdp.phi(str(this_state), observed_action) - del_theta_total_Q[this_state]) \
                                      * (exp_Q[this_state][observed_action] / sum_exp_Q[this_state])
                beta = exp_Q[this_state][observed_action]/sum_exp_Q[this_state]
                grad_wrt_theta += (1/beta) * del_beta_del_theta
        return grad_wrt_theta

    def gradientAscent(self, histories, theta_0=None, do_print=False, use_precomputed_phi=False):
        """
        @brief Performs Policy inference using Gradient Ascent from Section 7.2.1 of
        Sugiyama, 2015.
        """
        self.histories = histories
        if theta_0 == None:
            test_phi = self.mdp.phi(str(1), 'East')
            theta_0 = np.random.uniform(size=(test_phi.size, 1)).T
        self.mdp.theta = theta_0
        self.theta_size = [self.mdp.theta.size, 1]

        thresh = 0.05
        eps = 0.01
        iter_count = 0
        delta_theta_norm = np.inf
        # Loop until convergence
        while delta_theta_norm > thresh:
            iter_count += 1
            prev_theta = self.mdp.theta
            delJHat_wrt_theta = self.delHistDelThetaEst(self.mdp.theta, use_precomputed_phi)
            self.mdp.theta = self.mdp.theta + eps*delJHat_wrt_theta.T
            delta_theta_norm = np.linalg.norm(self.mdp.theta - prev_theta)
            if do_print:
                pprint('Iter#: {}, delta: {}'.format(iter_count, delta_theta_norm), indent=4)
        if do_print:
            pprint('Found Theta:')
            pprint(self.mdp.theta)
        exp_Q = {state: {act: np.exp(np.dot(self.mdp.theta, self.mdp.phi_at_state[state][act])) for act in
                         self.mdp.action_list} for state in self.mdp.state_vec}
        sum_exp_Q = {state: sum(exp_Q[state].values()) for state in self.mdp.state_vec}
        self.mdp.policy = {state: {act: exp_Q[int(state)][act]/sum_exp_Q[int(state)] for act in self.mdp.action_list}
                           for state in self.mdp.states}
        if do_print:
            print("Infered-Policy as a {state: action-distribution} dictionary.")
            pprint(self.mdp.policy)

    @staticmethod
    def evalGibbsPolicy(theta, phi, action, action_list):
        """
        @brief Returns an approximated policy update.

        @param theta vector of weights.
        @param phi vector of basis functions.
        @param action
        @param action_list
        """
        exp_Q = {act:np.exp(np.dot(theta, phi)) for act in action_list}

        return exp_Q[action]/sum(exp_Q.values())


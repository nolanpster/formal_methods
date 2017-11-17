#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

import numpy as np
from copy import deepcopy
from pprint import pprint

class MDP_solvers(object):
    """
    @brief Solves an MPD with a specified algorithm.

    @param mdp The mdp to solve.
    @param method a string matching the name of a method in @ref MDP_solvers.
    """
    def __init__(self, mdp=None, method=None):
        if mdp is not None:
            self.mdp = mdp
        if method is not None:
            self.setMethod(method)

    def solve(self, method=None, **kwargs):
        if method is not None and (not self.method == method):
            self.setMethod(method)
        # Unbound method call requires passing in 'self'.
        self.algorithm(self, **kwargs)
    def setMethod(self, method='valueIteration'):
            # Calles a method named 'method'. So self.algorithm points to the
            # method in the MDP_solvers class.
            self.method = method
            self.algorithm = getattr(MDP_solvers, method)

    def valueIteration(self, do_print=False):
        """
        @brief returns a dictiionary describing the policy: keys are states,
               values are actions.
        """
        # Follows procedure in section 7.4 of Jay Taylor's notes: 'Markov
        # Decision Processes: Lecture Notes for STP 425'.
        iter_count = 0 # Iteration counter
        epsilon = 0.01 # Stopping threshold
        # Initialize decision of each state to none.
        policy = {state: None for state in self.mdp.states}
        # Initialize value of each state to zero.
        values =  np.array([0.0 for _ in self.mdp.states])
        prev_values = np.array([float('inf') for _ in self.mdp.states])
        # Initial iteration check parameters.
        delta_value_norm = np.linalg.norm(values - prev_values)
        if self.mdp.gamma == 1.0:
            # Override gamma to be a very conservative discount factor.
            # This allows the solution to eventually converge if the problem is
            # well defined.
            gamma = 0.99999
        else:
            gamma = self.mdp.gamma
        thresh = epsilon*(1.0 - gamma) / (2.0*gamma)
        # Loop until convergence
        while delta_value_norm > thresh:
            iter_count += 1
            prev_values = deepcopy(values)
            for s_idx, state in enumerate(self.mdp.states):
                for act in self.mdp.action_list:
                    reward = self.mdp.reward[state][act]
                    # Column of Transition matrix
                    trans_prob = self.mdp.T(state, act)
                    # Using the actual discount factor, comput the value.
                    this_value = \
                             reward \
                             + self.mdp.gamma * np.dot(trans_prob, prev_values)
                    # Update value if new one is larger.
                    if this_value >= values[s_idx]:
                        values[s_idx] = this_value
                        policy[state] = act
            delta_value_norm = np.linalg.norm(values - prev_values)
        self.mdp.values = {state: value for state, value in \
                                                  zip(self.mdp.states, values)}
        self.mdp.policy = policy
        if do_print:
            print("Value iteration took {} iterations.".format(iter_count))
            pprint(self.mdp.values)
            print("Policy as a {state: action} dictionary.")
            pprint(self.mdp.policy)
        return

    def expectationMaximization(self, do_print=False):
        """
        @brief
        """

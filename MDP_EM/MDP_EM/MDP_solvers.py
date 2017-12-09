#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

import numpy as np
from copy import deepcopy
from pprint import pprint
from solution_video import SolutionVideo

class MDP_solvers(object):
    """
    @brief Solves an MPD with a specified algorithm.

    @param mdp The mdp to solve.
    @param method a string matching the name of a method in @ref MDP_solvers.
    """
    def __init__(self, mdp=None, method=None, write_video=False):
        if mdp is not None:
            self.mdp = mdp
        if method is not None:
            self.setMethod(method)
        self.write_video = write_video
        if self.write_video:
            self.video_writer = SolutionVideo(method, self.mdp.grid_map, self.mdp.stateRowColToNum,
                                              self.mdp.action_list)

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

    def valueIteration(self, do_print=False, policy_keys_to_print=None):
        """
        @brief returns a dictiionary describing the policy: keys are states,
               values are actions.
        """
        # Follows procedure in section 7.4 of Jay Taylor's notes: 'Markov
        # Decision Processes: Lecture Notes for STP 425'.
        iter_count = 0 # Iteration counter
        epsilon = 0.01 # Stopping threshold
        # Initialize decision of each state to none.
        empty_policy_dist = {act:0 for act in self.mdp.action_list}
        policy = {state: empty_policy_dist.copy() for state in self.mdp.states}
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
                    if this_value > values[s_idx]:
                        values[s_idx] = this_value
                        policy[state] = empty_policy_dist.copy()
                        policy[state][act] = 1
                    if self.write_video:
                        self.video_writer.render(policy)
            delta_value_norm = np.linalg.norm(values - prev_values)
        # Set zero-likly-hood states to take empty action.
        for state in self.mdp.states:
            if sum(policy[state].values()) == 0:
                policy[state][self.mdp.sink_act] = 1
        self.mdp.values = {state: value for state, value in \
                                                  zip(self.mdp.states, values)}
        self.mdp.policy = policy
        if do_print:
            print("Value iteration took {} iterations.".format(iter_count))
            pprint(self.mdp.values)
            print("Policy as a {state: action-distribution} dictionary.")
            if policy_keys_to_print is not None:
                policy_out  = {state: deepcopy(policy[state]) for state in policy_keys_to_print}
                pprint(policy_out)
            else:
                pprint(self.mdp.policy)
        return run_stats

    def expectationMaximization(self, do_print=False, policy_keys_to_print=None):
        """
        @brief
        """
        num_iters = 100
        S = deepcopy(self.mdp.S) # Initial distribution.
        for _ in range(num_iters):
            P = self.mdp.setProbMatGivenPolicy()
            R = [self.mdp.probRewardGivenX_T(state) for state in self.mdp.states]
            R = np.array(R)
            alpha, beta, P_R, P_T_given_R, expect_T_given_R = \
                MDP_solvers.e_step(self, S, R, P, self.mdp.gamma)
            MDP_solvers.m_step(self, beta)
            self.mdp.removeNaNValues()
            if self.write_video:
                self.video_writer.render(self.mdp.policy)
        if do_print:
            policy_out = deepcopy(self.mdp.policy)
            for state, act_dist in policy_out.items():
                for act, prob in act_dist.items():
                    policy_out[state][act] = round(prob,3)
            print("EM did {0} iterations in {1:.3f} seconds.".format(num_iters, elapsed_time))
            if policy_keys_to_print is not None:
                policy_out  = {state: policy_out[state] for state in policy_keys_to_print}
            print("EM found the following policy: {}")
            pprint(policy_out)
            print("Prob. or reward = {0:.3f}, expectation of T given R = "
                  "{1:.3f}.".format(P_R, expect_T_given_R))

    def e_step(self, S, R, P, gamma, H=100):
        """
        @brief Algorithm 1 from Toussaint and Storkey 2010.

        @param H Horizon length.
        """
        L = np.empty(2*H+1)
        _a = S
        _b = R
        alpha = _a
        beta = gamma*_b
        L[0] = np.inner(_a, _b)
        for h in range(1, H+1):
            _a = np.inner(P,_a)
            l_ind = 2*h-1
            L[l_ind] = gamma**(l_ind) * np.inner(_a, _b)
            l_ind += 1
            _b = np.inner(_b, P)
            L[l_ind] = gamma**(l_ind) * np.inner(_a, _b)
            alpha += gamma**h * _a
            beta += gamma**(h+1) * _b
        L *= (1-gamma)
        alpha *= (1-gamma)
        P_R = np.sum(L)
        P_T_given_R = L / P_R
        expect_T_given_R = np.dot(range(2*H+1), L) / P_R
        return alpha, beta, P_R, P_T_given_R, expect_T_given_R

    def m_step(self, beta):
        for state_ind, state in enumerate(self.mdp.states):
            norm_factor = 0
            # Update policy and record value in normalization factor.
            for act in self.mdp.action_list:
                self.mdp.policy[state][act] = self.mdp.policy[state][act] * \
                    (self.mdp.reward[state][act] +
                     np.inner(beta, self.mdp.prob[act][state_ind, :]))
                norm_factor += self.mdp.policy[state][act]
            if norm_factor > 0:
                for act in self.mdp.action_list:
                    self.mdp.policy[state][act] /= norm_factor

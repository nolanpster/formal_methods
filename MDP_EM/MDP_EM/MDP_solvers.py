#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

import numpy as np
from copy import deepcopy
from pprint import pprint
import time

#from solution_video import SolutionVideo

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
            self.video_writer = SolutionVideo(method, self.mdp.grid_map, self.mdp.gridCellRowColToNum,
                                              self.mdp.action_list)

    def solve(self, method=None, **kwargs):
        if method is not None and (not self.method == method):
            self.setMethod(method)
        # Unbound method call requires passing in 'self'.
        return self.algorithm(self, **kwargs)
    def setMethod(self, method='valueIteration'):
            # Calles a method named 'method'. So self.algorithm points to the
            # method in the MDP_solvers class.
            self.method = method
            self.algorithm = getattr(MDP_solvers, method)

    def valueIteration(self, do_print=False, policy_keys_to_print=None, print_iterations=False, **kwargs):
        """
        @brief returns a dictiionary describing the policy: keys are states,
               values are actions.
        """
        start_time = time.time()
        # Follows procedure in section 7.4 of Jay Taylor's notes: 'Markov
        # Decision Processes: Lecture Notes for STP 425'.
        iter_count = 0 # Iteration counter
        epsilon = 0.1 # Stopping threshold
        # Initialize decision of each state to none.
        empty_policy_dist = {act:0 for act in self.mdp.executable_action_dict[self.mdp.controllable_agent_idx]}
        policy = {state: empty_policy_dist.copy() for state in self.mdp.states}
        # Initialize value of each state to zero.
        values =  np.array([0.0 for _ in self.mdp.states])
        prev_values = np.array([float('inf') for _ in self.mdp.states])
        # Initial iteration check parameters.
        delta_value_norm = np.linalg.norm(values - prev_values)
        if self.mdp.gamma == 1.0:
            # Override gamma to be a very conservative discount factor.  This allows the solution to eventually converge
            # if the problem is well defined.
            gamma = 0.99999
        else:
            gamma = self.mdp.gamma
        thresh = epsilon
        # Loop until convergence
        while delta_value_norm > thresh:
            if do_print or print_iterations:
                iter_start_time = time.time()
            iter_count += 1
            prev_values = deepcopy(values)
            for s_idx, state in enumerate(self.mdp.states):
                for act in empty_policy_dist.keys():
                    this_value = self.mdp.reward[state][act]
                    # Column of Transition matrix
                    trans_prob = self.mdp.T(state, act)
                    # Using the actual discount factor, comput the value.
                    this_value += self.mdp.gamma * np.dot(trans_prob, prev_values)
                    # Update value if new one is larger.
                    if this_value > values[s_idx]:
                        values[s_idx] = this_value
                        policy[state] = empty_policy_dist.copy()
                        policy[state][act] = 1
            delta_value_norm = np.linalg.norm(np.subtract(values, prev_values))
            if do_print or print_iterations:
                print(" Change in state values: {:f}, {:f}sec".format(delta_value_norm, time.time() - iter_start_time))
        # Set zero-likly-hood states to take empty action.
        for state in self.mdp.states:
            if sum(policy[state].values()) == 0:
                policy[state][self.mdp.sink_action] = 1
        self.mdp.values = {state: value for state, value in zip(self.mdp.states, values)}
        self.mdp.policy = policy

        elapsed_time = time.time() - start_time
        run_stats = {'run_time': elapsed_time, 'iterations': iter_count}

        if do_print or print_iterations:
            print("Value iteration did {0} iterations in {1:.3f} seconds.".format(iter_count, elapsed_time))
        if do_print:
            print("State values:")
            pprint(self.mdp.values)
            print("Policy as a {state: action-distribution} dictionary.")
            if policy_keys_to_print is not None:
                policy_out  = {state: deepcopy(policy[state]) for state in policy_keys_to_print}
                pprint(policy_out)
            else:
                pprint(self.mdp.policy)
        return run_stats

    def expectationMaximization(self, do_print=False, policy_keys_to_print=None, horizon_length=25, num_iters=10,
                                do_incremental_e_step=False, print_iterations=False, **kwargs):
        """
        @brief
        """
        start_time = time.time()
        S = deepcopy(self.mdp.S) # Initial distribution.
        self.sink_act_distribution = {act:0 for act in self.mdp.executable_action_dict[self.mdp.controllable_agent_idx]}
        self.sink_act_distribution[self.mdp.sink_action] = 1.0

        # Precompute everything we can.
        if do_incremental_e_step:
            horizon_generator = xrange(1, horizon_length+1)
        self.mdp.buildEntireTransProbMat()

        for _iter in xrange(num_iters):
            if do_print or print_iterations:
                iter_start_time = time.time()
            self.policy_vec = self.mdp.getPolicyAsVec()
            self.policy_mat = self.policy_vec.reshape((self.mdp.num_states, self.mdp.num_executable_actions))
            P = self.mdp.setProbMatGivenPolicy(policy_mat=self.policy_mat)
            R = self.mdp.probRewardGivenX_T(use_reward_mat=True, policy_mat=self.policy_mat)
            if do_incremental_e_step:
                beta = self.mdp.gamma * R
                beta = self.incrementalEStep(beta, R, P, self.mdp.gamma, horizon_generator)
            else:
                alpha, beta, P_R, P_T_given_R, expect_T_given_R = MDP_solvers.e_step(self, S, R, P, self.mdp.gamma,
                                                                                     H=horizon_length)
            self.m_step(beta)
            if do_print or print_iterations and (not _iter % 25):
                print 'EM Iter:{} in {}sec'.format(_iter, time.time() - iter_start_time)
        elapsed_time = time.time() - start_time
        stats = {'run_time': elapsed_time, 'iterations': num_iters}
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
            if not do_incremental_e_step:
                # We calculated E{T|R} and P(R) along the way.
                print("Prob. or reward = {0:.3f}, expectation of T given R = "
                      "{1:.3f}.".format(P_R, expect_T_given_R))
        return stats

    def incrementalEStep(self, beta, R, P, gamma, horizon_generator):
        for h in horizon_generator:
            beta = np.multiply(gamma, np.add(R, np.inner(beta, P)))
        return beta

    def e_step(self, S, R, P, gamma, H=25):
        """
        @brief Algorithm 1 from Toussaint and Storkey 2010.

        Note that this is not optimized for speed. Many variables could be pre-computed, pre-allocated, and we could use
        something faster for matrix multiplication (e.g. einsum). If speed is an issue, use the incrementalEStep.

        @param H Horizon length.
        """
        L = np.empty(2*H+1)
        _a = deepcopy(S)
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
        expect_T_given_R = np.inner(range(2*H+1), L) / P_R
        return alpha, beta, P_R, P_T_given_R, expect_T_given_R

    def m_step(self, beta):
        beta_times_T_mat = np.einsum('j,ijk->ik', beta, self.mdp.trans_prob_mat)
        new_policy_mat = np.multiply(self.policy_mat,  (self.mdp.reward_mat + beta_times_T_mat))
        norm_factors = np.sum(new_policy_mat, axis=1)
        new_policy_mat = np.einsum('ij,i->ij', new_policy_mat, np.reciprocal(norm_factors))
        for state_ind, state in enumerate(self.mdp.states):
            # Update policy and record value in normalization factor.
            if norm_factors[state_ind] > 0:
                for act_ind, act in enumerate(self.mdp.executable_action_dict[self.mdp.controllable_agent_idx]):
                    self.mdp.policy[state][act] = new_policy_mat[state_ind, act_ind]
            else:
                # Zero policy, just pick sink_action.
                self.mdp.policy[state] = self.sink_act_distribution

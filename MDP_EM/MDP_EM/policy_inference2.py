#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

import numpy as np
from numpy.core.umath_tests import inner1d
from copy import deepcopy
from copy import copy
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import time
from collections import deque
import random
import warnings
import sys


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
        return self.algorithm(self, **kwargs)

    def setMethod(self, method='gradientAscent'):
            # Calles a method named 'method'. So self.algorithm points to the
            # method in the PolicyInference class.
            self.method = method
            if self.method is not None:
                self.algorithm = getattr(PolicyInference, method)
            else:
                self.algorithm = None

    def buildPolicy(self):
        """
        @brief Method to build the policy during/after policy inference.
        """
        if self.method is 'gradientAscent':
            exp_Q = {state: {act: np.exp(np.dot(self.mdp.theta, self.mdp.phi_at_state[state][act])) for act in
                             self.mdp.action_list} for state in self.mdp.state_vec}
            sum_exp_Q = {state: sum(exp_Q[state].values()) for state in self.mdp.state_vec}
            self.mdp.policy = {state: {act: exp_Q[int(state)][act]/sum_exp_Q[int(state)] for act in self.mdp.action_list}
                               for state in self.mdp.states}

    def gradientAscent(self, histories, theta_0=None, do_print=False, use_precomputed_phi=False, dtype=np.float64,
                       monte_carlo_size=None, reference_policy_vec=None, precomputed_observed_action_indeces=None):
        """
        @brief Performs Policy inference using Gradient Ascent from Section 7.2.1 of
               Sugiyama, 2015.

        @note All policy differences are computed with L1-norm.

        @param theta_0 Initial parameter vector used to start inference.
        @param do_print Flag to print policy difference and time for each iteration.
        @param use_procompute_phi Flag to use the self.mdp.phi_at_state state-action dictionary for inference.
        @param dtype Numpy data type to use for array computation, default is numpy.float64.
        @param [monte_carlo_size, Will repeat the inference <integer> times. If this input is supplied, the inference will
               return the policy inferred in the final run of the batch size, and arrays of policy differences and the
               time each iteration took to infer in seconds.
        @param reference_policy_vec] Used to compute policy difference for monte carlo, [num_states*num_actions] with
               actions in the same order as self.mdp.action_list.
        @param precomputed_observed_action_indeces If supplied, the inference will assume the correct observed action
               indeces for each time-step in each episode in the history. This is useful if the inference is being
               called externally with the same history.
        """
        def computePhis():
            # Initialize arrays for intermediate computations.
            phis = np.zeros([self.mdp.num_states, self.mdp.num_actions, theta_size], dtype=dtype)
            for state in self.mdp.state_vec:
                for act_idx, act in enumerate(acts_list):
                    for kern_idx in xrange(theta_size):
                        phis[state, act_idx, kern_idx] = self.mdp.phi_at_state[state][act][kern_idx]
            return phis

        
        # Process input arguments
        do_plot=False
        acts_list = self.mdp.action_list
        num_acts = self.mdp.num_actions
        num_states = self.mdp.num_states
        self.histories = histories
        (num_episodes, num_steps) = self.histories.shape
        traj_samples = range(num_episodes) # Will be shuffled every iteration.
        if monte_carlo_size is not None:
            batch_L1_norm = np.empty(monte_carlo_size)
            batch_infer_time = np.empty(monte_carlo_size)
            num_inferences = monte_carlo_size
            doing_monte_carlo = True
        else:
            num_inferences = 1
            doing_monte_carlo = False
        if precomputed_observed_action_indeces is not None:
            observed_action_indeces = precomputed_observed_action_indeces
        else:
            # Precompute observed actions for all episodes.
            observed_action_indeces = np.empty([num_episodes, num_steps], dtype=np.int8)
            for episode in xrange(num_episodes):
                for t_step in xrange(1, num_steps):
                    this_state = self.histories[episode, t_step-1]
                    next_state = self.histories[episode, t_step]
                    observed_action = self.mdp.graph.getObservedAction(this_state, next_state)
                    observed_action_indeces[episode, t_step] = acts_list.index(observed_action)

        # Initialize Weight vector, theta.
        if theta_0 == None:
            test_phi = self.mdp.phi(str(1), 'East')
            theta_0 = np.empty([test_phi.size, 1], dtype=dtype).T
            for kern_idx in xrange(self.mdp.num_kern):
                for act_idx, act in enumerate(self.mdp.action_list):
                        theta_0[0][kern_idx*self.mdp.num_actions+act_idx]= 1.0 / (theta_0.size)
        theta_size = theta_0.size

        phis = computePhis()

        # Velocity vector can be thought of as the momentum of the gradient descent. It is used to carry the theta
        # estimate through local minimums. https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/. Set at top of
        # for-loop.
        velocity_memory = 0.9

        # Configure printing and plotting options.
        vals2plot=[]
        del2print=[]

        # This block configures the iteration parameters. The threshold at which to stop iteration, @c thresh, the
        # fraction of gradient to apply in each iteration, @c eps, an iteration counter, and the change in the norm of
        # the average theta vector since the previous iteration.
        #
        # The stopping threshold for the Stochastic gradient ascent is based upon the the moving average  of the @c
        # theta vector as described here: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average. Take
        # the difference between the new average @theta and the previous @theta. If the Euclidian norm of the difference
        # is less than @c thresh, the iteration exits.
        thresh = 0.05
        eps = 0.25
        inverse_temp_start = np.float16(1.0)
        # Larger value of inverse_temp_rate causes the temperature to cool faster, reduces oscilation. Set to 0 to
        # remove effect of temperature cooling.
        inverse_temp_rate =  np.float16(0.25)
        
        sig=self.mdp.phi.std_devs
        for p in range(10):
            phis = computePhis()

            # For any calculations with numpy.einsum, unless otherwise noted:
            # - d : time-step axis
            # - h : garbage (length 1) axis
            # - i : state-axis
            # - j : action-axis
            # - k : theta/phi vector axis
            # - l : a policy represented as a vector
            for trial in xrange(num_inferences):
                is_last_trial = True if (num_inferences - trial)==1 else False
                trial_tic = time.clock()
                inverse_temp = inverse_temp_start
                iter_count = 0
                delta_theta_norm = np.inf
                velocity = np.zeros([theta_size], dtype=dtype)
                self.mdp.theta = deepcopy(theta_0)
                theta = self.mdp.theta # Use a local reference to theta for look-up speed.

                theta_avg = deepcopy(theta_0)

                # Loop until convergence
                while delta_theta_norm > thresh:
                    if do_print:
                        iter_tic = time.clock()
                    iter_count += 1
                    inverse_temp += inverse_temp_rate
                    temp = np.float16(1.0) / inverse_temp
                    prev_theta = copy(theta)
                    random.shuffle(traj_samples)
                    traj_queue = deque(traj_samples)

                    while len(traj_queue)>0:
                        episode = traj_queue.pop()

                        if use_precomputed_phi:
                            # Pre-compute all possible values (for small environments) using numpy.einsum where possible.
                            # The code below is equivalent to:
                            #   exp_Q = np.exp(np.sum(np.multiply(phis,theta[0]), axis=2))
                            #   sum_exp_Q = np.sum(exp_Q, axis=1)
                            #   for state in xrange(num_states):
                            #       sum_weighted_exp_Q[state] = np.dot(exp_Q[state], phis[state, :, :])
                            #   del_theta_total_Q = np.divide(sum_weighted_exp_Q.T, sum_exp_Q).T

                            exp_Q = np.exp(np.einsum('ijk,hk->ij', phis, theta))
                            reciprocal_sum_exp_Q = np.reciprocal(np.einsum('ij->i', exp_Q))
                            sum_weighted_exp_Q = np.einsum('ij,ijk->ki', exp_Q, phis)
                            # For this calc only because einsum uses a pedantic alphabetic convention:
                            # - i : phi axis
                            # - j : state axis
                            del_theta_total_Q = np.einsum('ij,j->ji', sum_weighted_exp_Q, reciprocal_sum_exp_Q)

                        else:
                            raise NotImplementedError

                        # Using Numpy.einsum, equivalent code is:
                        grad_wrt_theta = np.zeros(theta_size)
                        #   for t_step in xrange(1, num_steps):
                        #       this_state = histories[episode, t_step-1]
                        #       grad_wrt_theta += \
                        #           np.multiply(np.subtract(phis[this_state, observed_action_indeces[episode, t_step]],
                        #                                   del_theta_total_Q[this_state]), temp)
                        grad_wrt_theta = \
                            np.multiply(np.einsum('dk->k',
                                        np.subtract(phis[histories[episode,:-1], observed_action_indeces[episode,1:]],
                                                    del_theta_total_Q[histories[episode,:-1]])),
                                        temp)

                        velocity *= velocity_memory
                        velocity += np.multiply(eps, grad_wrt_theta.T)
                        theta += velocity


                    # Update moving average value of theta vector, then decrease the learning rate, @c eps.
                    theta_avg_old = copy(theta_avg)
                    theta_avg -= np.divide(theta_avg, iter_count);
                    theta_avg += np.divide(theta, iter_count);
                    vector_diff = np.subtract(theta_avg_old, theta_avg)
                    delta_theta_norm = np.einsum('ij->', np.absolute(vector_diff))

                    if do_plot:
                        vals2plot.append(self.mdp.theta.tolist())
                        del2print.append(delta_theta_norm)
                    if do_print:
                        infer_toc = time.clock() - iter_tic
                        pprint('Iter#: {}, delta: {}, iter-time: {} sec.'.format(iter_count, delta_theta_norm, infer_toc),
                               indent=4)

            def del_V_del_sig():
                del_sig=np.zeros(len(self.mdp.phi.std_devs))
                dpds=del_phi_del_sig()
                for state in self.mdp.state_vec:
                    for act_idx, act in enumerate(acts_list):
                        print(dpds.size)
                        del_sig=exp_Q[state,act_idx]*np.dot(dpds[state,act_idx,:],theta.T)/np.sum(exp_Q[state,:])-exp_Q[state,act_idx]*np.sum(exp_Q[state,:]*np.dot(dpds[state,act_idx,:],theta.T))

            def del_phi_del_sig(): 
                del_phis = np.zeros([self.mdp.num_states, theta_size], dtype=dtype)
                
                for state in self.mdp.state_vec:
                    for act_idx, act in enumerate(acts_list):
                        for kern_idx in range(theta_size):
                            del_phis[state,act_idx, kern_idx] = self.mdp.delPhi_delSig[kern_idx%(self.mdp.num_kern), state,act_idx]
                return del_phis

            sig=sig+del_V_del_sig()
            self.mdp.updateSigmas(sig)
            print('Sigmas={0} for iteration {1}'.format(sig,p))

        if do_print:
                pprint('Found Theta:')
                pprint(self.mdp.theta)

        if is_last_trial:
                self.buildPolicy()

        if doing_monte_carlo:
                batch_infer_time[trial] = time.clock() - trial_tic
                print('Infernce Batch Trial {} done in {} sec.'.format(trial+1, batch_infer_time[trial]))
                infered_policy_vec = np.einsum('ij,i->ij', exp_Q, reciprocal_sum_exp_Q)
                batch_L1_norm[trial] = np.einsum('l->', np.abs(np.subtract(infered_policy_vec.ravel(),
                                                                           reference_policy_vec)))

        if do_print:
                print("Infered-Policy as a {state: action-distribution} dictionary.")
                pprint(self.mdp.policy)

        if do_plot:
                plt.plot(range(iter_count),del2print)
                plt.show()

                for u in range(self.mdp.theta.size):
                    plt2.plot(range(iter_count),[vals2plot[o][0][u] for o in range(len(vals2plot))])
                    plt2.ylabel('Theta '+str(int(u/6))+'_'+str(u%6))
                    plt2.show()

            


        if doing_monte_carlo:
            return (batch_L1_norm, batch_infer_time)

    def historyMLE(self, histories, do_print=False):
        """
        @brief Given the set of demonstrated histories, use Maximum Likelihood Estimation to compute a tabular policy
               for each state.

        The Maximum Likelihood Estimator of the policy for each state is
            @c pi(state, act) = c(state, act) / [sum(c(state, act)) for all act in actions]
        Where c(state, act) is the numer of times `act` was observed in the input @ref histories. If an action is never
        taken, the resulting zero likelihood in the policy is replaced by @ref sys.float_info.min.

        @param histories A numpy array with a column listing the interger-valued state index at each time-step and a row
               for each desmonstation episode.
        @note If a state is not visited in the history, it will be given a policy of zeros.
        """
        self.histories = histories
        states_in_history = set(self.histories.ravel())
        if states_in_history != set(self.mdp.state_vec):
            warnings.warn('The following states were not visited in the history: {}. Their policies will be `nan` for '
                          'all actions.'.format(set(self.mdp.state_vec) - states_in_history))

        # Initialize decision of each state to take first action in MDP's @c action_list.
        empty_policy_dist = {act:np.array([[0.]]) for act in self.mdp.action_list}
        self.mdp.policy = {state: deepcopy(empty_policy_dist) for state in self.mdp.states}

        # For every state-action pair in the history, increment each observed action.
        (num_episodes, num_steps) = self.histories.shape
        for episode in xrange(num_episodes):
            for t_step in xrange(1, num_steps):
                this_state = self.histories[episode, t_step-1]
                next_state = self.histories[episode, t_step]
                observed_action = self.mdp.graph.getObservedAction(this_state, next_state)
                self.mdp.policy[str(this_state)][observed_action][0][0] += 1

        # Weight each action by the number of times the state was visited.
        for state in self.mdp.policy.keys():
            total_state_visits = float(np.sum(self.mdp.policy[state].values()))
            if total_state_visits > 0:
                for action in self.mdp.policy[state].keys():
                    if self.mdp.policy[state][action][0][0]==0:
                        # If the state was not visited, we need to give it the smalles non-zero float value so we can
                        # compute the KL-Divergence later. @todo - determine if this makes sense.
                        self.mdp.policy[state][action][0][0] = sys.float_info.min
                    else:
                        self.mdp.policy[state][action][0][0] /= total_state_visits

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
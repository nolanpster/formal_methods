#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

import numpy as np
from copy import deepcopy
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
        self.algorithm(self, **kwargs)

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

    def gradientAscent(self, histories, theta_0=None, do_print=False, use_precomputed_phi=False):
        """
        @brief Performs Policy inference using Gradient Ascent from Section 7.2.1 of
        Sugiyama, 2015.
        """

        # Process input arguments
        do_plot=False
        acts_list = self.mdp.action_list
        num_acts = self.mdp.num_actions
        num_states = self.mdp.num_states
        self.histories = histories
        (num_episodes, num_steps) = self.histories.shape
        traj_samples = range(num_episodes) # Will be shuffled every iteration.
        # Precompute observed actions for all episodes.
        observed_action_indeces = np.empty([num_episodes, num_steps], dtype=int)
        for episode in xrange(num_episodes):
            for t_step in xrange(1, num_steps):
                this_state = self.histories[episode, t_step-1]
                next_state = self.histories[episode, t_step]
                observed_action = self.mdp.graph.getObservedAction(this_state, next_state)
                observed_action_indeces[episode, t_step] = acts_list.index(observed_action)

        # Initialize Weight vector, theta.
        if theta_0 == None:
            test_phi = self.mdp.phi(str(1), 'East')
            theta_0 = np.empty([test_phi.size, 1]).T
            for kern_idx in xrange(self.mdp.num_kern):
                for act_idx, act in enumerate(self.mdp.action_list):
                        theta_0[0][kern_idx*self.mdp.num_actions+act_idx]= 1.0 / (theta_0.size)
        self.mdp.theta = theta_0
        theta_avg = deepcopy(theta_0)
        theta_size = self.mdp.theta.size

        # Velocity vector can be thought of as the momentum of the gradient descent. It is used to carry the theta
        # estimate through local minimums. https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/.
        velocity = np.zeros([theta_size])
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
        eps = 0.005
        iter_count = 0
        inverse_temp = 1.0
        # Larger value of inverse_temp_rate causes the temperature to cool faster, reduces oscilation. Set to 0 to
        # remove effect of temperature cooling.
        inverse_temp_rate =  1.0
        delta_theta_norm = np.inf
        theta = self.mdp.theta # Use a local reference to theta for look-up speed.

        # Initialize arrays for intermediate computations.
        phis = np.zeros([self.mdp.num_states, self.mdp.num_actions, theta_size])
        for state in self.mdp.state_vec:
            for act_idx, act in enumerate(acts_list):
                for kern_idx in xrange(theta_size):
                    phis[state, act_idx, kern_idx] = self.mdp.phi_at_state[state][act][kern_idx]
        phi_weighted_exp_Q = np.zeros(phis.shape)
        sum_weighted_exp_Q = np.empty([num_states, theta_size])

        # Loop until convergence
        while delta_theta_norm > thresh:
            if do_print:
                tic = time.clock()
            iter_count += 1
            inverse_temp += inverse_temp_rate
            temp = 1.0 / inverse_temp
            prev_theta = deepcopy(theta)
            random.shuffle(traj_samples)
            traj_queue = deque(traj_samples)

            while len(traj_queue)>0:
                episode = traj_queue.pop()

                if use_precomputed_phi:
                    # Pre-compute all possible values (for small environments).
                    exp_Q = np.exp(np.sum(phis * theta[0], axis=2))
                    sum_exp_Q = np.sum(exp_Q, axis=1)
                    for state in xrange(num_states):
                        sum_weighted_exp_Q[state] = np.dot(exp_Q[state],phis[state, :, :])
                    del_theta_total_Q = (sum_weighted_exp_Q.T/sum_exp_Q).T

                else:
                    raise NotImplementedError

                grad_wrt_theta = np.zeros(theta_size)
                # Note: This code does in-place operations for speed, apologies for decreasing the readability.
                for t_step in xrange(1, num_steps):
                    this_state = histories[episode, t_step-1]
                    grad_wrt_theta += (phis[this_state, observed_action_indeces[episode, t_step]]
                                       - del_theta_total_Q[this_state]) \
                                      * temp

                velocity *= velocity_memory
                velocity += eps*grad_wrt_theta.T
                theta += velocity

            # Update moving average value of theta vector, then decrease the learning rate, @c eps.
            theta_avg_old = deepcopy(theta_avg)
            theta_avg -= theta_avg / iter_count;
            theta_avg += theta / iter_count;
            delta_theta_norm = np.linalg.norm(theta_avg_old - theta_avg)

            if do_plot:
                vals2plot.append(self.mdp.theta.tolist())
                del2print.append(delta_theta_norm)
            if do_print:
                toc = time.clock() - tic
                pprint('Iter#: {}, delta: {}, iter-time: {}sec.'.format(iter_count, delta_theta_norm, toc), indent=4)

        if do_print:
            pprint('Found Theta:')
            pprint(self.mdp.theta)

        self.buildPolicy()
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

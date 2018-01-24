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


    def gradientAscent(self, histories, theta_0=None, do_print=False, use_precomputed_phi=False):
        """
        @brief Performs Policy inference using Gradient Ascent from Section 7.2.1 of
        Sugiyama, 2015.
        """

        # Process input arguments
        do_plot=False
        self.histories = histories
        states_in_history = set(self.histories.ravel())
        (num_episodes, num_steps) = self.histories.shape
        # Precompute observed actions for all episodes.
        observed_actions = dict.fromkeys(xrange(num_episodes), {})
        for episode in xrange(num_episodes):
            for t_step in xrange(1, num_steps):
                this_state = self.histories[episode, t_step-1]
                next_state = self.histories[episode, t_step]
                observed_actions[episode][t_step] = self.mdp.graph.getObservedAction(this_state, next_state)

        # Initialize Weight vector, theta.
        if theta_0 == None:
            test_phi = self.mdp.phi(str(1), 'East')
            theta_0 = np.random.uniform(-1, 1, size=(test_phi.size, 1)).T
        self.mdp.theta = theta_0
        theta_avg = deepcopy(theta_0)
        theta_size = self.mdp.theta.size

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
        delta_theta_norm = np.inf

        # Initialize arrays for intermediate computations.
        acts_lst = self.mdp.action_list
        phis = np.zeros([self.mdp.num_states, self.mdp.num_actions, theta_size])
        for state in self.mdp.state_vec:
            for act_idx, act in enumerate(self.mdp.action_list):
                for kern_idx in xrange(theta_size):
                    phis[state, act_idx, kern_idx] = self.mdp.phi_at_state[state][act][kern_idx]
        phi_weighted_exp_Q = np.zeros(phis.shape)

        # Loop until convergence
        while delta_theta_norm > thresh:
            if do_print:
                tic = time.clock()
            iter_count += 1
            prev_theta = deepcopy(self.mdp.theta)
            traj_samples = range(num_episodes)
            random.shuffle(traj_samples)
            traj_queue = deque(traj_samples)

            while len(traj_queue)>0:
                episode = traj_queue.pop()
                theta = self.mdp.theta

                if use_precomputed_phi:
                    # Pre-compute all possible values (for small environments).
                    exp_Q = np.exp(np.sum(phis * theta[0], axis=2))
                    sum_exp_Q = np.sum(exp_Q, axis=1)
                    for i in range(phis.shape[0]):
                        for j in xrange(phis.shape[1]):
                            phi_weighted_exp_Q[i,j] = phis[i,j]*exp_Q[i,j]
                    sum_weighted_exp_Q = np.sum(phi_weighted_exp_Q, axis=1).T
                    del_theta_total_Q = (sum_weighted_exp_Q/sum_exp_Q).T
                else:
                    raise NotImplementedError

                grad_wrt_theta = np.zeros(theta_size)
                # Note: This code does in-place operations for speed, apologies for decreasing the readability.
                for t_step in xrange(1, num_steps):
                    this_state = self.histories[episode, t_step-1]
                    this_act_idx = acts_lst.index(observed_actions[episode][t_step])

                    del_beta_del_theta = phis[this_state, this_act_idx]
                    del_beta_del_theta -= del_theta_total_Q[this_state]
                    del_beta_del_theta *= exp_Q[this_state, this_act_idx]
                    del_beta_del_theta /= sum_exp_Q[this_state]

                    reciprocal_beta = sum_exp_Q[this_state]
                    reciprocal_beta /= exp_Q[this_state, this_act_idx]
                    grad_wrt_theta += reciprocal_beta * del_beta_del_theta

                self.mdp.theta += eps*grad_wrt_theta.T

            # Update moving average value of theta vector, then decrease the learning rate, @c eps.
            theta_avg_old = deepcopy(theta_avg)
            theta_avg -= theta_avg / iter_count;
            theta_avg += self.mdp.theta / iter_count;
            delta_theta_norm = np.linalg.norm(theta_avg_old - theta_avg)
            eps *= 0.995

            if do_plot:
                vals2plot.append(self.mdp.theta.tolist())
                del2print.append(delta_theta_norm)
            if do_print:
                toc = time.clock() - tic
                pprint('Iter#: {}, delta: {}, iter-time: {}sec.'.format(iter_count, delta_theta_norm, toc), indent=4)

        if do_print:
            pprint('Found Theta:')
            pprint(self.mdp.theta)

        exp_Q = {state: {act: np.exp(np.dot(self.mdp.theta, self.mdp.phi_at_state[state][act])) for act in
                         self.mdp.action_list} for state in self.mdp.state_vec}
        sum_exp_Q = {state: sum(exp_Q[state].values()) for state in self.mdp.state_vec}
        self.mdp.policy = {state: {act: exp_Q[int(state)][act]/sum_exp_Q[int(state)] for act in self.mdp.action_list}
                           for state in self.mdp.states}
        print(self.mdp.theta)
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

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
import signal

import data_helper as DataHelp

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


class PolicyInference(object):
    """
    @brief Class that infers for polices using an instance of the MDP class.

    @pre Assumes the MDP instance has an instance variable .graph of type @ref GridGraph.
    """

    six_tabs = '\t'*6

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

    def getObservedActionIndeces(self):
        """
        @brief Precompute the observed action indeces for the instance's demonstration set.

        @return observed_action_indeces A Num_episodes-by-(num_steps) matrix of action indeces. Each column
                corresponds to the action taken from the state at time-step t from t=0 to T-1 so the first column is
                invalid.
        """
        (num_episodes, num_steps) = self.histories.shape
        observed_action_indeces = np.empty([num_episodes, num_steps],
                                           dtype=DataHelp.getSmallestNumpyUnsignedIntType(self.mdp.num_actions))
        for episode in xrange(num_episodes):
            for t_step in xrange(1, num_steps):
                this_state = self.histories[episode, t_step-1]
                next_state = self.histories[episode, t_step]
                observed_action = self.mdp.graph.getObservedAction(this_state, next_state)
                observed_action_indeces[episode, t_step] = self.mdp.action_list.index(observed_action)
        return observed_action_indeces

    def computePhis(self):
        # Initialize arrays for intermediate computations.
        phis = np.zeros([self.mdp.num_states, self.mdp.num_actions, self.theta_size], dtype=self.dtype)
        for state in self.mdp.grid_cell_vec:
            for act_idx, act in enumerate(self.mdp.action_list):
                for kern_idx in xrange(self.theta_size):
                    phis[state, act_idx, kern_idx] = self.mdp.phi_at_state[state][act][kern_idx]
        return phis

    def estimateHessian(self, theta, phis, exp_Q, sum_exp_Q, sum_weighted_exp_Q):
        """
        @brief This method computes the Hessian of the objective function, which is the gradient w.r.t. the parameter
               vector, theta.

        0) delLog(pi)_delTheta = phi - ( sum_over_actions(phi * exp(theta'*phi)) / sum_over_actions(exp(theta'*phi)) ).

        1) del/delTheta ( delLog(pi)_delTheta ) = del_delTheta( phi - (H / W) ) -- Note in CDC paper, H = lambda, W = z.

        2) del/delTheta ( delLog(pi)_delTheta ) =  - (W*delH_delTheta - H*delW_delTheta) / W^2
        """
        phis_self_outer_prod = np.empty([self.mdp.num_states, self.mdp.num_actions, self.theta_size, self.theta_size])
        sum_weighted_exp_Q_self_outer_prod = np.empty([self.mdp.num_states, self.theta_size, self.theta_size])
        for state in xrange(self.mdp.num_states):
            sum_weighted_exp_Q_self_outer_prod[state, :, :] = np.outer(sum_weighted_exp_Q[:, state],
                                                                       sum_weighted_exp_Q[:, state])
            for act_idx in xrange(self.mdp.num_actions):
                phis_self_outer_prod[state, act_idx, :, :] = np.outer(phis[state, act_idx, :], phis[state, act_idx, :])

        # See note in gradientAscent for einsum axis idences. Using 'l' for second phi axis. The lines below read:
        # a) delH/delTheta = sum_over_all_actions( exp(theta'*phi(s,a)) * outer(phi(s,a), phi(s,a)) )
        # b) W*delH/delTheta = sum_over_all_actions( exp(theta'*phi(s,a)) ) * delH/delTheta
        # c) H*delW/delTheta = outer-with-self of (sum_over_all_actions( exp(theta'*phi(s,a)) * phi(s,a) ) )
        delH_delTheta = np.einsum('ij, ijkl -> ikl', exp_Q, phis_self_outer_prod)
        W_prod_delH_delTheta = np.einsum('i, ikl -> ikl', sum_exp_Q, delH_delTheta)
        H_prod_delW_delTheta = sum_weighted_exp_Q_self_outer_prod
        W_squared_reciprocal = np.reciprocal(np.einsum('i, i -> i', sum_exp_Q, sum_exp_Q))

        # Note thta the leading minus sign aparent in step (2)  has already been applied:
        hessian_objective_numerator = -np.subtract(H_prod_delW_delTheta, W_prod_delH_delTheta)
        hessian_objective_func = np.einsum('ikl, i -> ikl', hessian_objective_numerator, W_squared_reciprocal)

        return hessian_objective_func

    def gradientAscent(self, histories, theta_0=None, do_print=False, use_precomputed_phi=False, dtype=np.float64,
                       monte_carlo_size=None, reference_policy_vec=None, precomputed_observed_action_indeces=None):
        """
        @brief Performs Policy inference using Gradient Ascent from Section 7.2.1 of
               Sugiyama, 2015.

        @note All policy differences are computed with L1-norm.
        @note For any calculations with numpy.einsum, unless otherwise noted:
                - d : time-step axis
                - h : garbage (length 1) axis
                - i : state-axis
                - j : action-axis
                - k : theta/phi vector axis
                - l : a policy represented as a vector

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
        # Process input arguments
        do_plot=False
        acts_list = self.mdp.action_list
        num_acts = self.mdp.num_actions
        num_states = self.mdp.num_states
        self.histories = histories
        (num_episodes, num_steps) = self.histories.shape
        self.dtype = dtype
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
            observed_action_indeces = self.getObservedActionIndeces()

        # Initialize Weight vector, theta.
        if theta_0 is None:
            test_phi = self.mdp.phi(1, 'East')
            theta_0 = np.empty([test_phi.size, 1], dtype=dtype).T
            for kern_idx in xrange(self.mdp.num_kern):
                for act_idx, act in enumerate(self.mdp.action_list):
                        theta_0[0][kern_idx*self.mdp.num_actions+act_idx]= 1.0 / (theta_0.size)
        self.theta_size = theta_0.size
        self.ones_length_theta = np.ones(self.theta_size)

        phis = self.computePhis()

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


        # For any calculations with numpy.einsum, unless otherwise noted:
        # - d : time-step axis
        # - h : garbage (length 1) axis
        # - i : state-axis
        # - j : action-axis
        # - k : theta/phi vector axis
        # - l : a policy represented as a vector
        # Loop until convergence unless killed by Crtl-C
        killer = GracefulKiller()
        for trial in xrange(num_inferences):
            # Loop until convergence
            if killer.kill_now:
                break
            is_last_trial = True if (num_inferences - trial)==1 else False
            trial_tic = time.clock()
            inverse_temp = inverse_temp_start
            iter_count = 0
            delta_theta_norm = np.inf
            velocity = np.zeros([self.theta_size], dtype=dtype)
            self.mdp.theta = deepcopy(theta_0)
            theta = self.mdp.theta # Use a local reference to theta for look-up speed.
            theta_avg = deepcopy(theta_0)

            while delta_theta_norm > thresh and not killer.kill_now:
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
                    #   for t_step in xrange(1, num_steps):
                    #       this_state = histories[episode, t_step-1]
                    #       grad_wrt_theta += \
                    #           np.multiply(np.subtract(phis[this_state, observed_action_indeces[episode, t_step]],
                    #                                   del_theta_total_Q[this_state]), temp)
                    grad_wrt_theta = np.einsum('dk->k',
                                               np.subtract(phis[histories[episode,:-1],
                                                           observed_action_indeces[episode,1:]],
                                               del_theta_total_Q[histories[episode,:-1]]))
                    grad_wrt_theta *= temp

                    velocity *= velocity_memory
                    velocity += np.multiply(eps, grad_wrt_theta)
                    theta += velocity

                # Update moving average value of theta vector, then decrease the learning rate, @c eps.
                theta_avg_old = copy(theta_avg)
                theta_avg -= np.divide(theta_avg, iter_count);
                theta_avg += np.divide(theta, iter_count);
                vector_diff = np.subtract(theta_avg_old, theta_avg)
                delta_theta_norm = inner1d(np.absolute(vector_diff), self.ones_length_theta)

                if do_plot:
                    vals2plot.append(self.mdp.theta.tolist())
                    del2print.append(delta_theta_norm)
                if do_print:
                    infer_toc = time.clock() - iter_tic
                    pprint('Iter#: {}, delta: {}, iter-time: {}sec.'.format(iter_count, delta_theta_norm, infer_toc),
                           indent=4)

            if do_print:
                pprint('Found Theta:')
                pprint(self.mdp.theta)

            estimate_covariance = False
            if estimate_covariance:
                # Following CDC paper's notation here: H is hessian, G is estimated auto-covariance of the gradient.

                # Sum over all observed transitions (d), and over all histories (h).
                # Maybe multiply by sqrt(N)?
                N = np.float32(histories.size)
                theta_gradient_per_step = np.subtract(phis[histories[:,:-1], observed_action_indeces[:,1:]],
                                                      del_theta_total_Q[histories[:,:-1]])

                # Precompute hessians at all states Size = [num_states, size_phi, size_phi].
                sum_exp_Q = np.reciprocal(reciprocal_sum_exp_Q)
                hessian_objective_func = self.estimateHessian(theta, phis, exp_Q, sum_exp_Q, sum_weighted_exp_Q)

                G_estimate = np.zeros([self.theta_size, self.theta_size])
                H_estimate = np.zeros([self.theta_size, self.theta_size])
                for traj in xrange(num_episodes):
                    for step in xrange(num_steps-1):
                        G_estimate += np.outer(theta_gradient_per_step[traj, step, :],
                                               theta_gradient_per_step[traj, step, :])
                        H_estimate += hessian_objective_func[histories[traj, step], :, :]
                G_estimate /= N
                H_estimate /= N
                H_est_inv = np.linalg.inv(H_estimate)
                parameter_variance = np.diag(np.dot(H_est_inv, np.dot(G_estimate, H_est_inv)))

            if is_last_trial:
                self.mdp.buildGibbsPolicy()

            if doing_monte_carlo:
                batch_infer_time[trial] = time.clock() - trial_tic
                print('Infernce Batch Trial {} done in {} sec.'.format(trial+1, batch_infer_time[trial]))
                infered_policy_vec = np.einsum('ij,i->ij', exp_Q, reciprocal_sum_exp_Q)
                batch_L1_norm[trial] = np.einsum('l->', np.abs(np.subtract(infered_policy_vec.ravel(),
                                                                           reference_policy_vec)))

            if do_print:
                print("Infered-Policy as a {state: action-distribution} dictionary.")
                self.mdp.buildGibbsPolicy()
                pprint(self.mdp.policy)

            if do_plot:
                plt.plot(range(iter_count),del2print)
                plt.show()

                for u in range(self.mdp.theta.size):
                    plt2.plot(range(iter_count),[vals2plot[o][0][u] for o in range(len(vals2plot))])
                    plt2.ylabel('Theta '+str(int(u/6))+'_'+str(u%6))
                    plt2.show()

        if killer.kill_now is True:
            print 'Search killed'
        if doing_monte_carlo:
            return (batch_L1_norm, batch_infer_time)

    def historyMLE(self, histories, do_print=False, do_weighted_update=False, reference_policy_vec=None):
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
        if states_in_history != set(self.mdp.grid_cell_vec):
            warnings.warn('The following states were not visited in the history: {}. Their policies will be `nan` for '
                          'all actions.'.format(set(self.mdp.grid_cell_vec) - states_in_history))

        # Initialize decision of each state to take first action in MDP's @c action_list.
        if do_weighted_update:
            prior_policy = deepcopy(self.mdp.policy)
        if not do_weighted_update:
            empty_policy_dist = {act:np.array([[0.]]) for act in self.mdp.action_list}
            self.mdp.policy = {state: deepcopy(empty_policy_dist) for state in self.mdp.states}

        # For every state-action pair in the history, increment each observed action.
        (num_episodes, num_steps) = self.histories.shape
        for episode in xrange(num_episodes):
            for t_step in xrange(1, num_steps):
                this_state = self.histories[episode, t_step-1]
                next_state = self.histories[episode, t_step]
                observed_action = self.mdp.graph.getObservedAction(this_state, next_state)
                if do_weighted_update:
                    action_weights = PolicyInference.actionProbGivenStatePair(this_state, next_state, prior_policy,
                                                                              self.mdp.P, self.mdp.action_list)
                    for act_idx, act in enumerate(self.mdp.action_list):
                        self.mdp.policy[this_state][act] += action_weights[act_idx]
                else:
                    self.mdp.policy[this_state][observed_action][0][0] += 1

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


    def iterativeBayes(self, histories, do_print=False, reference_policy_vec=None):
        self.histories = histories
        states_in_history = set(self.histories.ravel())
        if states_in_history != set(self.mdp.grid_cell_vec):
            warnings.warn('The following states were not visited in the history: {}. Their policies will be `nan` for '
                          'all actions.'.format(set(self.mdp.grid_cell_vec) - states_in_history))

        # Solve for initial guess of policy.
        self.historyMLE(histories, do_weighted_update=False)
        current_policy = self.mdp.getPolicyAsVec()
        # Improve the guess using the probability of an action given the observed states.
        for trial in xrange(10):
            self.historyMLE(histories, do_weighted_update=True)

    def buildPolicyVectors(self, phis, theta):
        """
        @brief Given Q(s,a) = <phi, theta>, this build the Boltzman policy as a vector.

        Vector indeces are [s_0_a_0, s_0_a_1, ... s_0_a_N, ... s_M_a_N-1, s_M_a_N] for M states and N actions.

        @param phis A Num-states--by--num-actions--by--num-kernels numpy array.
        @param theta A KxNum-kernels numpy array. Where K is the number of samples of the theta vector.

        @return policy_matrix  A matrix of action probabilities of shape K-by-(num-actions*num-states).
        """
        # See note about numpy.einsum axes in PolicyInference.gradientAscent(). In this case, 'h' is used to referce the
        # rows (number of samples) of theta.
        if 'gradientAscent' in self.method:
            exp_Q = np.exp(np.einsum('ijk,hk->hij', phis, theta))
            reciprocal_sum_exp_Q = np.reciprocal(np.einsum('hij->hi', exp_Q))
            policy_matrix = np.einsum('hij,hi->hij', exp_Q, reciprocal_sum_exp_Q)
            return policy_matrix.reshape(theta.shape[0], self.policy_vec_length)

    def logProbOfDataSet(self, theta_mat, phis):
        """
        @brief Given Q(s,a) = <phi(s,a), theta>, this evaluates the log probability of all trajectory given the
               parameter vector, theta.

        @param phis A Num-states--by--num-actions--by--num-kernels numpy array.
        @param theta A KxNum-kernels numpy array where K is the number of times theta is sampled.

        @return The log probability of all trajectorys for each theta sample.
        """
        policy_mat = self.buildPolicyVectors(phis, theta_mat)
        log_policy_mat = np.log(policy_mat)

        # Compute the log-likelihood of a trajectory given the sampled theta values.
        log_prob_traj_given_thetas = np.sum(log_policy_mat[:, self.episode_policy_vec_indeces], axis=1)

        return log_prob_traj_given_thetas

    def gradientAscentGaussianTheta(self, histories, theta_0=None, do_print=False, use_precomputed_phi=False,
                                    dtype=np.float64, monte_carlo_size=10, reference_policy_vec=None,
                                    precomputed_observed_action_indeces=None, theta_std_dev_0=None):
        """
        @brief Performs Policy inference using gradient ascent on the distribution theta_i ~ (mu_i, sigma_i).

        @note All policy differences are computed with L1-norm.
        @note For any calculations with numpy.einsum, unless otherwise noted:
                - d : time-step axis
                - h : sampled-theta axis
                - i : state-axis
                - j : action-axis
                - k : theta/phi vector axis
                - l : a policy represented as a vector

        @param histories A num-episodes - by - num-time-steps matrix of observed states.
        @param theta_0 Vector of initial theta distribution mean values.
        @param do_print Flag to print policy difference and time for each iteration.
        @param use_procompute_phi Flag to use the self.mdp.phi_at_state state-action dictionary for inference.
        @param dtype Numpy data type to use for array computation, default is numpy.float64.
        @param monte_carlo_size The number of times to sample from theta's distribution when performing monte-carlo
               integration of the log-likelihood of a demonstration set given theta's distribution.
        @param reference_policy_vec Not used by this implementation.
        @param precomputed_observed_action_indeces If supplied, the inference will assume the correct observed action
               indeces for each time-step in each episode in the history. This is useful if the inference is being
               called externally with the same history.
        """
        # Process input arguments
        do_plot=False
        acts_list = self.mdp.action_list
        num_acts = self.mdp.num_actions
        num_states = self.mdp.num_states
        self.monte_carlo_size = monte_carlo_size
        self.histories = histories
        (num_episodes, num_steps) = self.histories.shape
        self.dtype = dtype
        if precomputed_observed_action_indeces is not None:
            observed_action_indeces = precomputed_observed_action_indeces
        else:
            # Precompute observed actions for all episodes.
            observed_action_indeces = self.getObservedActionIndeces()

        # Initialize Weight vector, theta.
        if theta_0 is None:
            test_phi = self.mdp.phi(1, 'East')
            theta_0 = np.empty([test_phi.size, 1], dtype=dtype).T
            for kern_idx in xrange(self.mdp.num_kern):
                for act_idx, act in enumerate(self.mdp.action_list):
                    theta_0[0][kern_idx*self.mdp.num_actions+act_idx]= 1.0 / (theta_0.size)
        self.theta_size = theta_0.size
        self.ones_length_theta = np.ones(self.theta_size)
        self.mdp.theta = deepcopy(theta_0[0])
        theta_mean_vec = deepcopy(theta_0[0]) # Vector to compute SGD with.

        if theta_std_dev_0 is None:
            theta_std_dev_0 = np.ones(self.theta_size)
        theta_std_dev_vec = theta_std_dev_0 # Vector to compute SGD with.
        theta_std_dev_min = 0.04
        theta_std_dev_max = 10.

        # Precompute feature vector at all states.
        phis = self.computePhis()

        #  Precompute the indeces in a policy vector of length (num-states * num-actions) given the observed actions and
        #  the episodes. Note that the histories are (num-episodes by num-time-steps) large, but the matrix of policy
        #  vector indeces is (num-episodes by num-time-steps - 1).
        self.episode_policy_vec_indeces = (histories[:, :-1] * self.mdp.num_actions
                                           + observed_action_indeces[:, 1:]).ravel()
        self.policy_vec_length = self.mdp.num_actions * self.mdp.num_states

        # Velocity vector can be thought of as the momentum of the gradient descent. It is used to carry the theta
        # estimate through local minimums. https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/. Set at top of
        # for-loop.
        velocity_memory = 0.1
        velocity_mu = np.zeros([self.theta_size], dtype=dtype)
        velocity_sigma = np.zeros([self.theta_size], dtype=dtype)

        # Configure printing and plotting options.
        means2plot=np.empty(self.theta_size, dtype=dtype)
        sigmas2plot=np.empty(self.theta_size, dtype=dtype)
        del2print=[]

        # This block configures the iteration parameters. The threshold at which to stop iteration, @c thresh, the
        # fraction of gradient to apply in each iteration, @c eps, an iteration counter, and the change in the norm of
        # the average theta vector since the previous iteration.
        #
        # The stopping threshold for the Stochastic gradient ascent is based upon the the moving average  of the @c
        # theta vector as described here: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average. Take
        # the difference between the new average @theta and the previous @theta. If the Euclidian norm of the difference
        # is less than @c thresh, the iteration exits.
        theta_mu_avg = deepcopy(theta_0[0])
        theta_sigma_avg = deepcopy(theta_std_dev_0)
        max_log_prob_traj = np.log(0.8) * self.histories.size
        log_prob_thresh = np.log(0.999) * self.histories.size
        thresh = 0.05
        eps = 0.0001
        inverse_temp_start = np.float16(1.0)
        inverse_temp = inverse_temp_start
        # Larger value of inverse_temp_rate causes the temperature to cool faster, reduces oscilation. Set to 0 to
        # remove effect of temperature cooling.
        inverse_temp_rate =  np.float16(0.000)

        # Loop until convergence
        iter_count = 0
        theta_mu_old = np.inf
        theta_sigma_old = np.inf
        delta_theta_mu_norm = np.inf
        delta_theta_sigma_norm = np.inf
        log_prob_traj_given_mean_thetas  = -np.inf
        # Loop until convergence unless killed by Crtl-C
        killer = GracefulKiller()
        while (log_prob_traj_given_mean_thetas < (log_prob_thresh + max_log_prob_traj)) and not killer.kill_now:
            if do_print:
                iter_tic = time.clock()
            iter_count += 1
            inverse_temp += inverse_temp_rate
            temp = np.float16(1.0) / inverse_temp

            # Sample monte_carlo_size theta vectors from their distributions. This forms a
            # monte_carlo_size-by-theta_size matrix.
            theta_samples= np.random.multivariate_normal(theta_mean_vec, np.diag(theta_std_dev_vec),
                                                         self.monte_carlo_size)

            log_prob_traj_given_thetas = self.logProbOfDataSet(theta_samples, phis)
            ## Mu Update ##
            # Calculate the gradient of the log-likelihood of the sampled thetas with respect to the means of the
            # theta distribution.
            theta_variance = np.power(theta_std_dev_vec, 2)
            theta_sample_less_mean = theta_samples
            theta_sample_less_mean -= theta_mean_vec
            grad_log_prob_theta_wrt_mu = theta_sample_less_mean
            grad_log_prob_theta_wrt_mu /= theta_variance

            # Calculate the gradient of the log-likelihood of the trajectory with respect to the means of the theta
            # distribution.
            grad_log_prob_hist_given_theta_dist_wrt_mu = np.einsum('hk,h->k', grad_log_prob_theta_wrt_mu,
                                                                   log_prob_traj_given_thetas)
            grad_log_prob_hist_given_theta_dist_wrt_mu /= monte_carlo_size
            grad_log_prob_hist_given_theta_dist_wrt_mu *= temp

            # Update the gradient with the velocity of theta_mu.
            velocity_mu *= velocity_memory
            velocity_mu += np.multiply(eps, grad_log_prob_hist_given_theta_dist_wrt_mu)
            theta_mean_vec += velocity_mu

            ## Sigma Update ##
            # Calculate the gradient of the log-likelihood of the sampled thetas with respect to the standard
            # deviations of the theta distribution.
            grad_log_prob_theta_wrt_sigma = np.power(theta_sample_less_mean, 2)
            grad_log_prob_theta_wrt_sigma -= theta_variance
            grad_log_prob_theta_wrt_sigma /= np.multiply(theta_variance, theta_std_dev_vec) # Sigma^3

            # Calculate the gradient of the log-likelihood of the trajectory with respect to the standard deviations
            # of the theta distribution.
            grad_log_prob_hist_given_theta_dist_wrt_sigma = np.einsum('hk,h->k', grad_log_prob_theta_wrt_sigma,
                                                                      log_prob_traj_given_thetas)
            grad_log_prob_hist_given_theta_dist_wrt_sigma /= monte_carlo_size
            grad_log_prob_hist_given_theta_dist_wrt_sigma *= temp

            # Update the gradient with the velocity of theta_std_devs
            velocity_sigma *= velocity_memory
            velocity_sigma += np.multiply(eps, grad_log_prob_hist_given_theta_dist_wrt_sigma)
            theta_std_dev_vec += velocity_sigma

            # Check for any invalid standard deviations.
            theta_std_dev_vec[theta_std_dev_vec < theta_std_dev_min] = theta_std_dev_min
            theta_std_dev_vec[theta_std_dev_vec > theta_std_dev_max] = theta_std_dev_max

            # Update moving average value of theta distribution average.
            theta_mu_avg_old = copy(theta_mu_avg)
            theta_mu_avg -= np.divide(theta_mu_avg, iter_count);
            theta_mu_avg += np.divide(theta_mean_vec, iter_count);
            vector_diff = np.subtract(theta_mu_avg_old, theta_mu_avg)
            delta_theta_mu_norm = inner1d(np.absolute(vector_diff), self.ones_length_theta)

            # Update moving average value of theta distribution standard deviation.
            theta_sigma_avg_old = copy(theta_sigma_avg)
            theta_sigma_avg -= np.divide(theta_sigma_avg, iter_count);
            theta_sigma_avg += np.divide(theta_std_dev_vec, iter_count);
            vector_diff = np.subtract(theta_sigma_avg_old, theta_sigma_avg)
            delta_theta_sigma_norm = inner1d(np.absolute(vector_diff), self.ones_length_theta)

            log_prob_traj_given_mean_thetas = self.logProbOfDataSet(np.expand_dims(theta_mean_vec,axis=0), phis)

            if do_plot:
                means2plot = np.vstack((means2plot, theta_mean_vec))
                sigmas2plot = np.vstack((sigmas2plot, theta_std_dev_vec))
            if do_print:
                infer_toc = time.clock() - iter_tic
                pprint('Iter#: {}, delta_mu: {}, delta_sigma: {}, mean_LogLike: {}, iter-time: {}sec.'
                       .format(iter_count, delta_theta_mu_norm, delta_theta_sigma_norm, log_prob_traj_given_mean_thetas,
                               infer_toc),
                       indent=4)
                if not iter_count % 10:
                    print(PolicyInference.six_tabs + ' Max Log-likelihood: {}.'
                          .format(log_prob_traj_given_thetas.max()))

        # Prepare to exit.
        self.mdp.theta = np.expand_dims(theta_mean_vec, axis=0)

        if do_print:
            pprint('Found Theta:')
            pprint(self.mdp.theta)
            pprint('Covariance:')
            pprint(theta_std_dev_vec)

        self.mdp.buildGibbsPolicy()

        if do_print:
            print("Infered-Policy as a {state: action-distribution} dictionary.")
            pprint(self.mdp.policy)

        self.buildPolicyUncertainty(theta_std_dev_vec, phis, do_print)
        self.mdp.policy_uncertainty_as_vec = self.mdp.getPolicyAsVec(policy_to_convert=self.mdp.policy_uncertainty)

        if do_plot:
            repeated_indeces = np.repeat(np.expand_dims(range(self.mdp.theta.size), 0), iter_count, 0).T
            repeated_indeces = np.repeat(np.expand_dims(range(iter_count), 0), self.theta_size, 0).T
            plt.figure()
            plt.plot(repeated_indeces, means2plot[1:, :])
            plt.figure()
            plt.plot(repeated_indeces, sigmas2plot[1:, :])
            plt.show()

        if killer.kill_now is True:
            print 'Search killed'
        self.mdp.theta = theta_mean_vec
        self.mdp.theta_std_dev = theta_std_dev_vec
        return theta_mean_vec

    def buildPolicyUncertainty(self, theta_std_dev, phis, do_print=False):
        empty_policy_dist = {act:np.array([[0.]]) for act in self.mdp.action_list}
        policy_uncertainty = {state: deepcopy(empty_policy_dist) for state in self.mdp.states}
        for state in xrange(self.mdp.num_states):
            for act_idx, action in enumerate(self.mdp.action_list):
                policy_uncertainty[state][action] = np.sqrt(np.dot(np.power(theta_std_dev,2),
                                                                   np.power(phis[state, act_idx],2)))
        if do_print:
            print('Policy Uncertainty')
            pprint(policy_uncertainty)
        self.mdp.policy_uncertainty = policy_uncertainty

    @staticmethod
    def actionProbGivenStatePair(s_0, s_1, policy, trans_prob_func, action_list):
        """
        @param policy A policy in dictionary representation used by MDP.py.
        @param trans_prob_func A reference to the MDP.P() method.
        """
        total_prob_of_s_1 = np.sum([trans_prob_func(s_0, act, s_1) for act in action_list])
        policy_at_state = np.array([policy[s_0][act][0][0] for act in action_list])
        transition_prob_to_s_1 = np.array([trans_prob_func(s_0, act, s_1) for act in action_list])

        return (transition_prob_to_s_1 * policy_at_state) / total_prob_of_s_1

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

#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'
from policy_inference import PolicyInference
from feature_vector import FeatureVector
from copy import deepcopy
from MDP import MDP
import numpy as np


class ProductMDPxDRA(MDP):
    """A Markov Decision Process, defined by an initial state, transition model --- the probability transition matrix,
    np.array prob[a][0,1] -- the probability of going from 0 to 1 with action a.  and reward function. We also keep
    track of a gamma value, for use by algorithms. The transition model is represented somewhat differently from the
    text.  Instead of T(s, a, s') being probability number for each state/action/state triplet, we instead have T(s, a)
    return a list of (p, s') pairs.  We also keep track of the possible states, terminal states, and actions for each
    state.  The input transitions is a dictionary: (state,action): list of next state and probability tuple.  AP: a set
    of atomic propositions. Each proposition is identified by an index between 0 -N.  L: the labeling function,
    implemented as a dictionary: state: a subset of AP."""
    def __init__(self, mdp, dra, sink_action=None, sink_list=[]):
        """
        @brief
        """
        # Set self to be an MDP instance. Build the product ontop of this, then call setSinks again now that the object
        # is fully populated.
        super(self.__class__, self).__init__()
        self.computeProductMDPxDRA(mdp, dra)
        self.grid_map = deepcopy(mdp.grid_map)
        self.sink_action = sink_action
        self.reconfigureConditionalInitialValues()
        self.setSinks(sink_list)

    def computeProductMDPxDRA(self, mdp, dra):
        # Create product MDP-times-DRA
        if mdp.init is not None:
            init = (mdp.init, dra.get_transition(mdp.L[mdp.init], dra.initial_state))
        else:
            init = tuple()
        if mdp.init_set is not None:
            self.init_set = [(m_i, dra.get_transition(mdp.L[m_i], dra.initial_state)) for m_i in mdp.init_set]
        self.init = init
        prod_states = []
        for _s in mdp.states:
            for _q in dra.states:
                prod_states.append((_s, _q))
        N = len(prod_states)
        self.action_list = list(mdp.action_list)
        self.states = list(prod_states)
        for act in self.action_list:
            self.prob[act] = np.zeros((N, N))
            for prod_state in range(N):
                (_s,_q) = self.states[prod_state]
                self.L[(_s, _q)] = mdp.L[_s]
                for _j in range(N):
                    (next_s,next_q) = self.states[_j]
                    if next_q == dra.get_transition(mdp.L[next_s], _q):
                        _p = mdp.P(_s, act, next_s)
                        self.prob[act][prod_state, _j] =  _p
        mdp_acc = []
        for (J,K) in dra.acc:
            Jmdp = set([])
            Kmdp = set([])
            for prod_state in prod_states:
                if prod_state[1] in J:
                    Jmdp.add(prod_state)
                if prod_state[1] in K:
                    Kmdp.add(prod_state)
            mdp_acc.append((Jmdp, Kmdp))
        self.acc = mdp_acc
        self.dra = deepcopy(dra)

    def step(self):
        """
        @brief Given the current state and the policy, creates the joint distribution of next-states and actions. Then
               samples from that distribution.

        Returns the current state number as an integer.
        """
        #@TODO make work for all agents
        # Creates a transition probability vector of the same dimesion as a row in the transition probability matrix.
        this_trans_prob = np.zeros(self.num_states)
        this_policy = self.policy[self.current_state]
        for act in this_policy.keys():
            this_trans_prob += this_policy[act] * self.prob[act][self.states.index(self.current_state), :]
        # Renormalize distribution - need to deal with this in a better way.
        this_trans_prob /= this_trans_prob.sum()
        # Sample a new state given joint distribution of states and actions.
        try:
            next_index= np.random.choice(self.num_states, 1, p=this_trans_prob)[0]
        except ValueError:
            import pdb; pdb.set_trace()
        self.current_state = self.states[next_index]
        return self.current_state[0]

    def resetState(self):
        """
        @brief Reset state to a random position in the grid.
        """
        if self.init_set is not None:
            self.current_state = self.states[np.random.choice(self.num_states, 1, p=self.S)[0]]
        elif self.init is not None:
            self.current_state = self.init
        return self.current_state[0]

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
        # cell_state_slicer: used to extract indeces from the tuple of states. The joint-state tuples are used as
        # dictionary keys and this class augments the state tuple from (mdp_state) to ((mdp_state),dra_state).
        self.state_slice_length = 1
        self.cell_state_slicer = slice(None, self.state_slice_length, None)
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

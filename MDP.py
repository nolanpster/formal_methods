#!/usr/bin/env python
__author__ = 'Jie Fu, jfu2@wpi.edu'
from NFA_DFA_Module.NFA import NFA
from MDP_solvers import MDP_solvers

from scipy import stats
import numpy as np


class MDP:
    """A Markov Decision Process, defined by an initial state,
        transition model --- the probability transition matrix, np.array
        prob[a][0,1] -- the probability of going from 0 to 1 with action a.
        and reward function. We also keep track of a gamma value, for
        use by algorithms. The transition model is represented
        somewhat differently from the text.  Instead of T(s, a, s')
        being probability number for each state/action/state triplet,
        we instead have T(s, a) return a list of (p, s') pairs.  We
        also keep track of the possible states, terminal states, and
        actions for each state.  The input transitions is a
        dictionary: (state,action): list of next state and probability
        tuple.  AP: a set of atomic propositions. Each proposition is
        identified by an index between 0 -N.  L: the labeling
        function, implemented as a dictionary: state: a subset of AP."""
    def __init__(self, init=None, action_list=[], states=[], prob=dict([]),
                 acc=None, gamma=.9, AP=set([]), L=dict([]), reward=dict([])):
        self.init=init # Initial state
        self.action_list=action_list
        self.num_actions = len(self.action_list)
        self.states=states
        self.state_vec = np.array(map(int, self.states))
        self.num_states = len(self.states)
        self.current_state = None
        self.acc=acc
        self.gamma=gamma
        self.reward=reward
        self.prob=prob
        self.AP=AP # Atomic propositions
        self.L=L # Labels of states
        self.S = None # Initial probability distribution.
        # For EM Solving
        self.act_ind = lambda a_0, a_i: int(a_0 == a_i)
        if self.num_actions > 0:
            self.makeUniformPolicy()
        self.init_set = None
        self.setInitialProbDist()

    def T(self, state, action):
        """
        Transition model.  From a state and an action, return a row in the
        matrix for next-state probability.
        """
        i=self.states.index(state)
        return self.prob[action][i, :]

    def P(self, state, action, next_state):
        """
        Derived from the transition model. For a state, an action and the
        next_state, return the probability of this transition.
        """
        i=self.states.index(state)
        j=self.states.index(next_state)
        return self.prob[action][i, j]

    def getActions(self, state):
        S=set([])
        for _a in self.action_list:
            if not np.array_equal(self.T(state,_a), np.zeros(self.num_states)):
                S.add(_a)
        return S

    def labeling(self, s, A):
        self.L[s]=A

    def sample(self, state, action, num=1):
        """
        Sample the next state according to the current state, the action, and
        the transition probability.
        """
        if action not in self.getActions(state):
            return None # Todo: considering adding the sink state
        i=self.states.index(state)
        # Note that only one element is chosen from the array, which is the
        # output by random.choice
        next_index= np.random.choice(self.num_states, num, p=self.prob[action][i,:])[0]
        return self.states[next_index]

    def resetState(self):
        """
        @brief Reset state to a random position in the grid.
        """
        if self.init_set is not None:
            self.current_state = self.states[np.random.choice(self.num_states, 1,
                                                              p=self.S)[0]]
        elif self.init is not None:
            self.current_state = self.init
        return int(self.current_state[0])

    def addKernels(self, kernels):
        self.kernels = kernels
        self.num_kern = len(self.kernels)

    def phi(self, state, action):
        # Create vector of basis functions, phi. All kernels are multiplied an
        # action indicator function. A feature vector will have @c m*p
        # elements, where @c m is the number of actions, and @c p is the number
        # of kernels. This function takes arguments (<str>state, <str>action).
        phi = np.empty([self.num_actions*self.num_kern, 1]) # Column vector
        i_state = int(state)
        for _i, act in enumerate(self.action_list):
            this_ind = lambda a_in, a_i=act: self.act_ind(a_in, a_i)
            trans_probs = self.T(state, act)
            for _j, kern in enumerate(self.kernels):
                # Eq. 3.3 Sugiyama 2015
                try:
                    kern_weights = np.array(map(kern, self.state_vec))
                    phi[_i+(_j)*self.num_actions] = \
                        this_ind(action) * np.inner(trans_probs, kern_weights)
                except:
                    import pdb; pdb.set_trace()
        return phi

    def timePrior(self, _t):
        """
        Returns the geometric time prior from [TS2010?] for a finite-time MDP,
        @c P(T) = (1-gamma)*gamma^t.
        """
        return (1-self.gamma) * self.gamma**_t

    def probRewardGivenX_T(self, state, policy=None):
        """
        @brief The probability of a reward given a final state x at final time
        T, and a policy.

        If no policy is provided, then this defaults to mdp.policy.
        """
        if policy is None:
            policy = self.policy
        # List rewards available at this state for every action.
        prob_reward = 0
        for act in self.action_list:
            prob_reward += self.reward[state][act]*self.policy[state][act]

        return prob_reward

    def setProbMatGivenPolicy(self, policy=None):
        """
        @brief Returns a transition probability matrix that has been summed over all actions
        multiplied with all transition probabilities.

        Rows of un-reachable states are not guranteed to sum to 1.
        """
        if policy is None:
            policy = self.policy
        # Assume that all transition probability matricies are the same size.
        # The method below should work in python 2 and 3.
        prob_keys = tuple(self.prob.keys())
        self.prob_mat_given_policy = np.zeros_like(self.prob[prob_keys[0]])

        for state_idx, state_str in enumerate(self.states):
            this_policy = self.policy[state_str]
            for act in this_policy.keys():
                self.prob_mat_given_policy[state_idx,:] += \
                    this_policy[act]*self.prob[act][state_idx, :]
        return self.prob_mat_given_policy

    def setInitialProbDist(self, initial_state=None):
        # S based on section 1.2.2 of Toussaint and Storkey - the initial
        # distribution.
        if initial_state is None:
            # Default to uniform distribution.
            self.S = np.ones(self.num_states)/self.num_states
        elif type(initial_state) is list:
            init_prob = 1.0/len(initial_state)
            self.S = np.zeros(self.num_states)
            for state in initial_state:
                state_idx = self.states.index(state)
                self.S[state_idx] = init_prob
        else:
            self.S = np.zeros(self.num_states)
            init_idx = self.states.index(self.init)
            self.S[init_idx] = 1

    def makeUniformPolicy(self):
        uninform_prob = 1.0/self.num_actions
        uniform_policy_dist = {act: uninform_prob for act in self.action_list}
        self.policy = {state: uniform_policy_dist.copy() for state in
                       self.states}

    def setSinks(self, sink_frag):
        """
        @brief Finds augmented states that contain @c sink_frag and updates
               the row corresponding to their transition probabilities so that
               all transitions take a self loop with probability 1.

        Used for product MDPs (see @ref productMDP), to identify augmented
        states that include @c sink_frag. @c sink_frag is the DRA/DFA
        component of an augmented state that is terminal.
        """
        for state in self.states:
            if sink_frag in state:
                # Set the transition probability of this state to always self
                # loop.
                s_idx = self.states.index(state)
                for act in self.action_list:
                    self.prob[act][s_idx, :] = np.zeros((1, self.num_states))
                    self.prob[act][s_idx, s_idx] = 1.0

    def removeNaNValues(self):
        for state, action_dict in self.policy.items():
            for act, prob in action_dict.items():
                if np.isnan(prob):
                    self.policy[state][act] = 0

    @staticmethod
    def productMDP(mdp, dra):
        pmdp=MDP()
        if mdp.init is not None:
            init=(mdp.init, dra.get_transition(mdp.L[mdp.init],
                                               dra.initial_state))
        states=[]
        for _s in mdp.states:
            for _q in dra.states:
                states.append((_s, _q))
        N=len(states)
        pmdp.init=init
        pmdp.action_list=list(mdp.action_list)
        pmdp.states=list(states)
        for _a in pmdp.action_list:
            pmdp.prob[_a]=np.zeros((N, N))
            for i in range(N):
                (_s,_q)=pmdp.states[i]
                pmdp.L[(_s,_q)]=mdp.L[_s]
                for _j in range(N):
                    (next_s,next_q)=pmdp.states[_j]
                    if next_q == dra.get_transition(mdp.L[next_s], _q):
                        _p=mdp.P(_s,_a,next_s)
                        pmdp.prob[_a][i, _j]= _p
        mdp_acc=[]
        for (J,K) in dra.acc:
            Jmdp=set([])
            Kmdp=set([])
            for _s in states:
                if _s[1] in J:
                    Jmdp.add(_s)
                if _s[1] in K:
                    Kmdp.add(_s)
            mdp_acc.append((Jmdp, Kmdp))
        pmdp.acc=mdp_acc
        pmdp.num_states = len(pmdp.states)
        pmdp.num_actions = len(pmdp.action_list)
        pmdp.makeUniformPolicy()
        if pmdp.num_states > 0 and pmdp.init is not None:
            pmdp.setInitialProbDist(pmdp.init)
        return pmdp


    @staticmethod
    def get_NFA(mdp):
        """
        This function obtains the graph structure, which is essentially an
        non-deterministic finite state automaton from the original mdp.
        """
        nfa=NFA()
        nfa.initial_state=mdp.init
        nfa.states=mdp.states
        nfa.alphabet=mdp.action_list
        for _a in mdp.action_list:
            for s in mdp.states:
                next_state_list=[]
                for next_s in mdp.states:
                    if mdp.prob[_a][mdp.states.index(s), mdp.states.index(next_s)] != 0:
                        next_state_list.append(next_s)
                nfa.add_transition(_a, s, next_state_list)
        nfa.final_states=mdp.terminals
        return nfa

    @staticmethod
    def sub_MDP(mdp, H):
        """
        For a given MDP and a subset of the states H, construct a sub-mdp
        that only includes the set of states in H, and a sink states for
        all transitions to and from a state outside H.
        """
        if H == set(mdp.states):
            # If H is the set of states in mdp, return mdp as it is.
            return mdp
        submdp=MDP()
        submdp.states=list(H)
        submdp.states.append(-1) # -1 is the sink state.
        N=len(submdp.states)
        submdp.action_list=list(mdp.action_list)
        submdp.prob={_a:np.zeros((N, N)) for _a in submdp.action_list}
        temp=np.zeros(len(mdp.states))
        for k in set(mdp.states) - H:
            temp[mdp.states.index(k)]=1
        for _a in submdp.action_list:
            for s in H: # except the last sink state.
                i=submdp.states.index(s)
                for next_s in H:
                    j=submdp.states.index(next_s)
                    submdp.prob[_a][i, j] = mdp.P(s, _a, next_s)
                submdp.prob[_a][i, -1]= np.inner(mdp.T(s, _a), temp)
            submdp.prob[_a][submdp.states.index(-1), submdp.states.index(-1)]=1
        acc=[]
        for (J,K) in mdp.acc:
            Jsub = set(H).intersection(J)
            Ksub = set(H).intersection(K)
            acc.append((Jsub,Ksub))
        acc.append(({}, {-1}))
        submdp.acc = acc
        return submdp


    @staticmethod
    def read_from_file_MDP(fname):
        """
        This function takes the input file and construct an MDP based on thei
        transition relations. The first line of the file is the list of states.
        The second line of the file is the list of actions.
        Starting from the second line, we have
        state, action, next_state, probability
        """
        f=open(fname, 'r')
        array = []
        for line in f:
            array.append( line.strip('\n') )
        f.close()
        mdp=MDP()
        state_str=array[0].split(",")
        mdp.states=[int(i) for i in state_str]
        act_str=array[1].split(",")
        mdp.action_list=act_str
        mdp.prob=dict([])
        N=len(mdp.states)
        for _a in mdp.action_list:
            mdp.prob[_a]=np.zeros((N, N))
        for line in array[2: len(array)]:
            trans_str=line.split(",")
            state=int(trans_str[0])
            act=trans_str[1]
            next_state=int(trans_str[2])
            p=float(trans_str[3])
            mdp.prob[act][mdp.states.index(state), mdp.states.index(next_state)]=p
        return mdp

    def solve(self, method='valueIteration', **kwargs):
        """
        @brief Solves a given MDP. Defaults to the Value Iteration method.

        @param an instance of @ref MDP.
        @param a string matching a method name in @ref MDP_solvers.py.
        """
        MDP_solvers(self, method=method).solve(**kwargs)

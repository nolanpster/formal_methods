#!/usr/bin/env python
__author__ = 'Jie Fu, jfu2@wpi.edu, Nolan Poulin, nipoulin@wpi.edu'
from NFA_DFA_Module.NFA import NFA
from MDP_solvers import MDP_solvers
from policy_inference import PolicyInference

from scipy import stats
from copy import deepcopy
import numpy as np
from pprint import pprint
import sys # For float_info.epsilon.


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
    def __init__(self, init=None, action_list=[], states=[], prob=dict([]), acc=None, gamma=.9, AP=set([]), L=dict([]),
                 reward=dict([]), grid_map=None, act_prob=dict([])):
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
        self.grid_map=grid_map
        self.act_prob_mat_row_idx = dict.fromkeys(self.states)
        self.precomputeStateActProbMatRows()
        self.neighbor_dict = None
        self.prob=prob
        if any(prob):
            self.prob=prob
        elif any(act_prob):
            # Build it now Assume format of act_prob is {act: [rows(location-class) x cols(act_list_order)]}.
            self.act_prob = act_prob
            self.buildProbDict()
        self.AP=AP # Atomic propositions
        self.L=L # Labels of states
        self.S = None # Initial probability distribution.
        # For EM Solving
        self.act_ind = lambda a_0, a_i: int(a_0 == a_i)
        if self.num_actions > 0:
            self.makeUniformPolicy()
        self.init_set = None
        self.sink_act = None
        self.sink_list = []
        self.setInitialProbDist()
        # GridGraph class is used for path searching. Assigned externally.
        self.graph = None
        self.theta = None

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

    def stateRowColToNum(self,state):
        return self.grid_map[state[0], state[1]]


    def getActProbMatRow(self, state):
        """
        @brief Returns a row index of the Transition probability matrices corresponding to the current state.

        @state an array [row,col]
        """
        grid_row, grid_col = np.where(self.grid_map==state)
        act_prob_mat_row = 0
        if grid_row == 0:
            if grid_col == 0:
                # North West corner
                act_prob_mat_row = 6
            elif grid_col == self.grid_map.shape[1]-1:
                # North East corner
                act_prob_mat_row = 5
            else:
                # On North Wall.
                act_prob_mat_row = 1
        elif grid_row == self.grid_map.shape[0]-1:
            if grid_col == 0:
                # South West corner
                act_prob_mat_row = 8
            elif grid_col == self.grid_map.shape[1]-1:
                # South East corner
                act_prob_mat_row = 7
            else:
                # On South Wall
                act_prob_mat_row = 2
        elif grid_col == 0:
            # On West Wall
            act_prob_mat_row = 4
        elif grid_col == self.grid_map.shape[1]-1:
            # On East wall
            act_prob_mat_row = 3
        else:
            # In open space.
            pass
        return act_prob_mat_row

    def precomputeStateActProbMatRows(self):
        """
        @brief see title.
        """
        for state in self.act_prob_mat_row_idx.keys():
            self.act_prob_mat_row_idx[state] = self.getActProbMatRow(int(state))

    def buildNeighborDict(self):
        self.neighbor_dict = {} # {this_state: next_state: prob}
        for state_0 in self.states:
            self.neighbor_dict[state_0] = {}
            (this_row, this_col) = np.where(self.grid_map==int(state_0))
            # ID valid actions from this state.
            this_act_prob_mat_row_idx = self.act_prob_mat_row_idx[state_0]
            valid_acts = ['Empty']
            if this_act_prob_mat_row_idx == 0:
                valid_acts = self.action_list
            elif this_act_prob_mat_row_idx == 1:
                valid_acts += ['South', 'East', 'West']
            elif this_act_prob_mat_row_idx == 2:
                valid_acts += ['North', 'East', 'West']
            elif this_act_prob_mat_row_idx == 3:
                valid_acts += ['North', 'South', 'West']
            elif this_act_prob_mat_row_idx == 4:
                valid_acts += ['North', 'South', 'East']
            elif this_act_prob_mat_row_idx == 5:
                valid_acts += ['South', 'West']
            elif this_act_prob_mat_row_idx == 6:
                valid_acts += ['South', 'East']
            elif this_act_prob_mat_row_idx == 7:
                valid_acts += ['North', 'West']
            elif this_act_prob_mat_row_idx == 8:
                valid_acts += ['North', 'East']

            # +/- Correspond to "row/col" motions for cardinal directions.
            if 'North' in valid_acts:
                next_row = this_row - 1
                self.neighbor_dict[state_0][self.stateRowColToNum(np.concatenate((next_row, this_col)))] = 'North'
            if 'South' in valid_acts:
                next_row = this_row + 1
                self.neighbor_dict[state_0][self.stateRowColToNum(np.concatenate((next_row, this_col)))] = 'South'
            if 'East' in valid_acts:
                next_col = this_col + 1
                self.neighbor_dict[state_0][self.stateRowColToNum(np.concatenate((this_row, next_col)))] = 'East'
            if 'West' in valid_acts:
                next_col = this_col - 1
                self.neighbor_dict[state_0][self.stateRowColToNum(np.concatenate((this_row, next_col)))] = 'West'
            self.neighbor_dict[state_0][self.stateRowColToNum(np.concatenate((this_row, this_col)))] = 'Empty'


    def buildProbDict(self):
        """
        @brief Given a dictionary of action probabilities, create the true prob mat dictionary structure.
        """
        self.prob = {}
        if self.neighbor_dict is None:
            self.buildNeighborDict()
        for act in self.action_list:
            self.prob[act]=np.zeros((self.num_states, self.num_states))
            for state in self.state_vec:
                for next_state, ideal_act in self.neighbor_dict[str(state)].iteritems():
                        self.prob[act][state, next_state]= self.act_prob[act][self.act_prob_mat_row_idx[str(state)],
                                                                              self.action_list.index(ideal_act)]


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

    def step(self):
        """
        @brief Given the current state and the policy, creates the joint distribution of
            next-states and actions. Then samples from that distribution.

        Returns the current state number as an integer.
        """
        # Creates a transition probability vector of the same dimesion as a row in the
        # transition probability matrix.
        this_trans_prob = np.zeros(self.num_states)
        this_policy = self.policy[self.current_state]
        for act in this_policy.keys():
            this_trans_prob += this_policy[act] \
                               * self.prob[act][self.states.index(self.current_state), :]
        # Renormalize distribution - need to deal with this in a better way.
        this_trans_prob /= this_trans_prob.sum()
        # Sample a new state given joint distribution of states and actions.
        try:
            next_index= np.random.choice(self.num_states, 1, p=this_trans_prob)[0]
        except ValueError:
            import pdb; pdb.set_trace()
        self.current_state = self.states[next_index]
        return int(self.current_state[0])

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
        phi = np.zeros([self.num_actions*self.num_kern, 1]) # Column vector
        i_state = int(state)
        for _i, act in enumerate(self.action_list):
            this_ind = lambda a_in, a_i=act: self.act_ind(a_in, a_i)
            if this_ind(action):
                trans_probs = self.T(state, act)
                for _j, kern in enumerate(self.kernels):
                    # Eq. 3.3 Sugiyama 2015
                    kern_weights = np.array(map(kern, self.state_vec))
                    phi[_i+(_j)*self.num_actions] = \
                        this_ind(action) * np.inner(trans_probs, kern_weights)
        return phi

    def precomputePhiAtState(self):
        self.phi_at_state = {state: {act: self.phi(str(state), act) for act in self.action_list} for state in
                             self.state_vec}

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
                self.sink_list.append(state)
                s_idx = self.states.index(state)
                for act in self.action_list:
                    self.prob[act][s_idx, :] = np.zeros((1, self.num_states))
                    self.prob[act][s_idx, s_idx] = 1.0

    def removeNaNValues(self):
        """
        @brief Find NaN entries in policy, and replace them with a 0 or 1.

        Only replace with 1 if the state is in the list of sink states and the action
        equals the "sink action".
        """
        for state, action_dict in self.policy.items():
            this_total_prob = 0
            for act, prob in action_dict.items():
                if np.isnan(prob):
                    if state in self.sink_list and act==self.sink_act:
                        for zero_act in self.action_list:
                            if not(zero_act==act):
                                self.policy[state][zero_act] = 0
                            else:
                                self.policy[state][act] = 1
                        this_total_prob = 1
                    else:
                        self.policy[state][act] = 0
                else:
                    this_total_prob += prob
            if this_total_prob == 0:
                # Zero policy, just pick sink_action.
                self.policy[state][self.sink_act] = 1
            elif this_total_prob > 1.0+sys.float_info.epsilon:
                pass
                #import pdb; pdb.set_trace()
                #raise ValueError('Total probability greater than 1!')

    def computeKLDivergenceOfPolicyFromHistories(self, histories):
        """
        @brief Computes the KL-Divergence of a policy that should generate trajectories given the set of trajectories.

        Provided a set of histories that has a row for every trajectory, this computes the likelihood of each
        trajectory p(tau|D), given that it is known to be part of the history-set, and the probability of the trajectory
        given the policy of the MDP, p(tau|theta). It returns the KL-Divergence of p(tau|theta) from p(tau|D) summed
        over all trajectories.

        @pre The policy must be inferred/solved for.
        """
        (num_episodes, num_steps) = histories.shape

        # Compute the likelihood of a trajectory in the history data-set.
        # From https://stackoverflow.com/questions/27000092/count-how-many-times-each-row-is-present-in-numpy-array
        histories_data_type = np.dtype((np.void, histories.dtype.itemsize * histories.shape[1]))
        contiguous_histories = np.ascontiguousarray(histories).view(histories_data_type)
        unique_episodes, episode_count = np.unique(contiguous_histories, return_counts=True)
        unique_episodes = unique_episodes.view(histories.dtype).reshape(-1, histories.shape[1])
        num_unique_episodes = len(episode_count)
        episode_freq = episode_count / float(num_episodes)


        # Start probability of traj_given_policy as probability of state_0, initial distribution `S`.
        prob_of_traj_given_policy = deepcopy(self.S[[unique_episodes[:,0]]])
        for episode in xrange(num_unique_episodes):
            for t_step in xrange(2, num_steps):
                this_state = unique_episodes[episode, t_step-1]
                next_state = unique_episodes[episode, t_step]
                observed_action = self.graph.getObservedAction(this_state, next_state)
                prob_of_traj_given_policy[episode] *= self.P(str(this_state), observed_action, str(next_state))
                prob_of_traj_given_policy[episode] *= self.policy[str(this_state)][observed_action]
        return np.sum(np.multiply(episode_freq, np.log(np.divide(episode_freq, prob_of_traj_given_policy))))

    def getPolicyAsVec(self, policy_keys_to_use=None):
        """
        @brief returns a numpy array representing the policy  with actions listed in the same order as self.action_list.

        @param policy_keys_to_use If supplied, policy vector will include only the states listed. Additionally, this
               assumes that the state structure is that of the MDPxDRA, ('grid_stat_num', 'qX'), and it will extract the
               first element of the tuple into the intermediate dictionary.
        """
        if policy_keys_to_use is not None:
            policy_vec = np.empty(len(policy_keys_to_use) * self.num_actions)
            policy_dict  = {state[0]: deepcopy(self.policy[state]) for state in policy_keys_to_use}
        else:
            policy_vec = np.empty(self.num_states * self.num_actions)
            policy_dict = self.policy
        for state in xrange(len(policy_vec) / self.num_actions):
            for act_idx, act in enumerate(self.action_list):
                policy_vec[state * self.num_actions+act_idx] = policy_dict[str(state)][act]
        return policy_vec

    @staticmethod
    def productMDP(mdp, dra):
        pmdp=deepcopy(MDP())
        if mdp.init is not None:
            init=(mdp.init, dra.get_transition(mdp.L[mdp.init],
                                               dra.initial_state))
        if mdp.init_set is not None:
            pmdp.init_set = [(m_i, dra.get_transition(mdp.L[m_i], dra.initial_state))
                             for m_i in mdp.init_set]
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
        pmdp.dra = deepcopy(dra)
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

    @staticmethod
    def comparePolicies(reference_policy, comparison_policy, policy_keys_to_print, compare_to_decimals=3,
                        do_print=True, compare_policy_has_extra_keys=True, compute_kl_divergence=False,
                        reference_policy_has_augmented_states=True, compare_policy_has_augmented_states=False):
        # Use compare_policy_has_extra_keys=False for infered policy format.
        copied_ref_policy  = {state: deepcopy(reference_policy[state]) for state in policy_keys_to_print}
        if compare_policy_has_extra_keys:
            copied_compare_policy  = {state: deepcopy(comparison_policy[state]) for state in policy_keys_to_print}
        else:
            copied_compare_policy = deepcopy(comparison_policy)
        # Compute KL Divergence before values in copied policies are rounded.
        if compute_kl_divergence:
            policy_kl_divergence = MDP.computePolicyKLDivergence(
                    copied_ref_policy, copied_compare_policy,
                    reference_policy_has_augmented_states=reference_policy_has_augmented_states,
                    compare_policy_has_augmented_states=compare_policy_has_augmented_states)
        else:
            policy_kl_divergence = None
        # Create a copy of the dict whose values are to be overwritten by the differences.
        policy_difference = deepcopy(copied_ref_policy)
        for state, action_dict in copied_ref_policy.items():
            for act in action_dict.keys():
                copied_ref_prob = round(copied_ref_policy[state][act], compare_to_decimals)
                copied_ref_policy[state][act] = copied_ref_prob
                if compare_policy_has_extra_keys:
                    compare_state = state
                else:
                    compare_state = state[0]
                copied_compare_prob = round(copied_compare_policy[compare_state][act], compare_to_decimals)
                copied_compare_policy[compare_state][act] = copied_compare_prob
                policy_difference[state][act] = round(abs(copied_ref_prob - copied_compare_prob),
                                                      compare_to_decimals)
        if do_print:
            print("Policy Difference:")
            pprint(policy_difference)
            print("KL-Divergence of the comparison policy from the reference policy =  "
                  "{:.03f}.".format(policy_kl_divergence))
        return policy_difference, policy_kl_divergence

    @staticmethod
    def computePolicyKLDivergence(reference_policy, comparison_policy, reference_policy_has_augmented_states=False,
                                  compare_policy_has_augmented_states=False):
        """
        @brief Compute the KL Divergence between two policies, as expressed here:
               http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf.

        @param reference_policy The true distribution of the data.
        @param comparison_policy The learned distribution of the data.
        @param reference_policy_has_augmented_states A flag that should be `True` if the keys in the reference policy
               are in the form `(<int>state, dra_state)`, False expectes the keys are a `state` integer.

        @note Ensure that policies have not been rounded! KL-Divergence is undefined for zero likelihood actions.
        """
        reference_list = []
        compare_list = []
        for state, action_dict in reference_policy.items():
            for act in action_dict.keys():
                reference_list.append(reference_policy[state][act])
                if not reference_policy_has_augmented_states or \
                    (compare_policy_has_augmented_states and reference_policy_has_augmented_states):
                    compare_state = state
                else:
                    compare_state = state[0]
                compare_list.append(comparison_policy[compare_state][act])
        reference_vec = np.squeeze(np.array(reference_list))
        compare_vec = np.squeeze(np.array(compare_list)).T # Transpose to make both row vectors.
        with np.errstate(divide='ignore'):
            log_vec = np.log(reference_vec / compare_vec)
        # Catch computation errors caused by zeros in the reference and comparison vectors. After the`log` computation,
        # all values equal to `-np.inf` were the result of a zero-likelihood action in the reference policy. For
        # elements where both of these statements are true, we can substitute 0=0*log(0), proven by convergence.
        # Elements that are valued `np.nan` are the result of division by zero, and we can only substitute these values
        # with zero if both the reference and comparison policy elements have zero likelihood. @todo Ensure that
        # `np.nan` values are only located in un-reachable or sink states in the comparison policy (for EM-solved
        # policy).
        neg_inf_quotient = log_vec==-np.inf
        ref_zero_likelihood_action = reference_vec==0
        comp_zero_likelihood_action = compare_vec==0
        log_vec[neg_inf_quotient & ref_zero_likelihood_action] = 0.
        log_vec[ref_zero_likelihood_action & comp_zero_likelihood_action] = 0.

        kl_divergence = np.sum(reference_vec * log_vec)
        return kl_divergence


    def solve(self, method='valueIteration', write_video=False, **kwargs):
        """
        @brief Solves a given MDP. Defaults to the Value Iteration method.

        @param an instance of @ref MDP.
        @param a string matching a method name in @ref MDP_solvers.py.
        """
        MDP_solvers(self, method=method, write_video=write_video).solve(**kwargs)

    def inferPolicy(self, method='gradientAscent', write_video=False, **kwargs):
        """
        @brief Infers the policy of a given MDP. Defaults to the Gradient Ascent method.

        @param an instance of @ref MDP.
        @param a string matching a method name in @ref policy_solvers.py.
        """
        return PolicyInference(self, method=method, write_video=write_video).infer(**kwargs)

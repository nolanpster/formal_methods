#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

import itertools

from NFA_DFA_Module.DFA import DFA


class TS(DFA):
    """
    @brief This is a Transition system (TS).
    """
    default_config = {'state_variables': set([]),
                      'actions': set([]),
                      'state_transitions': {},
                      'atom_prop' : set([]),
                      'initial_state': set([]),
                      'final_states': None,
                      'current_state': None,
                      'labeled_states': None}

    @staticmethod
    def evalProposition(atom_prop, state):
        true_prop = set([])
        for prop in atom_prop:
            true_prop |= set(prop).intersection(state)
        return true_prop

    @staticmethod
    def powerSet(s):
        """
        @brief From: http://salvia.logdown.com/posts/249530-hello-world
        """
        powerset = set()
        for i in xrange(2**len(s)):
            subset = tuple([x for j,x in enumerate(s) if (i >> j) & 1])
            powerset.add(subset)
        return powerset

    @staticmethod
    def str2Tup(st):
        """
        @brief Format an input string to work with set notation create in
               @ref powerSet.
        """
        return tuple([st])

    def  __init__(self, state_variables=None, actions=None,
                  state_transitions=None, initial_state=None, atom_prop=None):
        """
        @brief Construct an instance of a @ref TS.

        @param states a set of states.
        @param actions a list of actions.
        @param state_transitions a subset of S*Act*S, is a transition relation.
        @param initial_state a state in S that the @ref TS starts from.
        @param atom_prop a set of atomic propositions. Per
        @param label the labeling function that gives each state a label that
               is in the powerset of atomic propositions.
        """
        # Create a set from each list, and then cast the set as a list. This
        # removes duplicates and sets are useful for mathematics.
        #
        # Interpret States
        if state_variables is not None:
            self.state_variables = state_variables
        else:
            self.state_variables = TS.default_config['state_variables']
        self.states = TS.powerSet(self.state_variables)
        self.num_states = len(self.states)
        # Interpret actions
        if actions is not None:
            if type(actions) is set:
                self.actions = actions
            elif type(actions) is list:
                self.actions = {TS.str2Tup(a) for a in actions}
        # Interpret atomic propositions.
        if atom_prop is not None:
            if type(atom_prop) is set:
                self.atom_prop = atom_prop
            elif type(atom_prop) is list:
                self.atom_prop = {TS.str2Tup(ap) for ap in atom_prop}
        else:
            self.atom_prop = TS.default_config['atom_prop']
        # Interpret initial state
        if initial_state is not None:
            if type(initial_state) is not set:
                initial_state = TS.str2Tup(initial_state)
            if initial_state in self.states:
               self.initial_state = initial_state
               self.current_state = self.initial_state
            else:
               raise ValueError("Provided initial states must be a subset of "
                                "the input set of states.")
        else:
            self.initial_state = TS.default_config['initial_state']
        # Assume user input a dictioary of state_transitions formated for a
        # @ref DFA.
        if state_transitions is not None:
            self.state_transitions = state_transitions
        else:
            self.state_transitions = TS.default_config['state_transitions']
        # Initialize any remaining instance variables with the default class
        # variables.
        for key, value  in TS.default_config.iteritems():
            if not self.__dict__.has_key(key):
                self.__dict__[key] = value

    def __del__(self):
        """
        @brief Destroy an instance of @ref TS
        """

    def reset(self):
        """
        @brief Reset the state machine to a default configuration.
        """
        self.current_state = self.initial_state

    def addAction(self, act):
        if type(act) is str:
            act = TS.str2Tup(act)
        self.actions.add(act)

    def doAction(self, act):
        """
        @brief Perform the desired action on the @ref TS.
        """
        if type(act) is str:
            act = TS.str2Tup(act)
        if act in self.actions:
            this_transition = tuple((act, self.current_state))
            if this_transition in self.state_transitions.keys():
                self.current_state = self.state_transitions[this_transition]
            else:
                raise ValueError("The transition {} from {} does not exist."
                                 .format(str(act),str(self.current_state)))
        else:
            raise ValueError("The requested action: {} does not exist."
                             .format(str(act)))

    def addState(self, state):
        """
        @state a string of a new state.
        """
        if state not in self.states:
            self.state_variables.add(state)
            self.states = TS.powerSet(self.state_variables)
            self.num_states = len(self.states)

    def add_transition(self, action, state, next_state=None):
        """
        Name convention matches NFA_DFA_Module
        @param action string
        @param state string
        @param (optional) string
        """
        # Parse and update actions, if necessary.
        if type(action) is str:
            action = TS.str2Tup(action)
        if action in self.actions:
            pass
        else:
            self.addAction(action)
        # Parse and update states, if necessary.
        if type(state) is str:
            state = TS.str2Tup(state)
        if type(state) is tuple or list:
            state = self.checkForMatchingState(state)
        # Parse and update next state and transitions.
        if next_state is None:
            next_state = state
        else:
            if type(next_state) is str:
                next_state = TS.str2Tup(next_state)
            if type(next_state) is tuple or list:
                next_state = self.checkForMatchingState(next_state)
        # This implementation only works for Deterministic transitions.
        self.state_transitions[(action, state)] = next_state

    def transitionsFrom(self, state):
        next_states = []
        actions = []
        for act in self.actions:
            next_state = self.get_transition(act, state)
            if next_state is not None:
                next_states.append(next_state)
                actions.append(act)
        return zip(itertools.repeat(state), actions, next_states)

    def labelStates(self):
        self.labeled_states = {}
        for state in self.states:
            self.labeled_states[state] = TS.evalProposition(self.atom_prop,
                                                            state);
    def getCurrentLabel(self):
        if self.labeled_states is None:
            self.labelStates()
        return self.labeled_states[self.current_state]


    def checkForMatchingState(self, state):
        # Look for equivalent state but need to convert tuples to sets.
        state = tuple(state) #redunant if already tuple
        found = False
        for known_state in self.states:
            if set(known_state) == set(state):
                state = known_state
                found = True
                break
        if not found:
            self.addState(state[0]) #only add string to the state_variables.
        return state

    def toDot(self,filename):
        with open(filename,'w') as dotfile:
            dotfile.write("digraph G { rankdir = LR\n")
            i = 0
            indexed_states = {}
            if self.labeled_states is None:
                self.labelStates()
            for state, label in self.labeled_states.items():
                indexed_states[state]=i
                # get rid of nuisance characters.
                state = ', '.join(state)
                label = ', '.join(label)
                dotfile.write("\t"+str(i)
                              + "[label=\"" + "labeled: {" + label
                              + "}\\n state: <"  + state + ">"
                              + "\",shape=box]\n")
                i += 1
            for ((a,s0),s1) in self.state_transitions.items():
                a = ''.join(a) # remove nuisance charachters
                dotfile.write("\t"+str(indexed_states[s0])
                        + "->"
                        + str(indexed_states[s1])
                        + "[label = \""
                        + str(a)
                        + "\"]\n")
            dotfile.write("}")

#!/usr/bin/env python
__author__ = 'Jie Fu, jfu2@wpi.edu'

from types import *

class ExceptionFSM(Exception):

    """This is the FSM Exception class."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class LTL_plus(str):
    """
    @brief Base class for a LTL_plus in alphabet.
    """
    # Harded Coded as empty string
    empty = 'E'

    def __new__(cls, value, *args, **kwargs):
        return super(LTL_plus, cls).__new__(cls, value)

    def __invert__(self):
        return  LTL_plus(''.join(("not_",self)))

    def __add__(self,other):
        return LTL_plus(' '.join((self, other)))

    def __radd__(self,other):
        return LTL_plus(' '.join((other, self)))

    def Or(self, other, label=None):
        if type(other) is not LTL_plus:
            raise ValueError('Type of {} must be LTL_plus.'
                             .format(other))
        if other == ~self:
            # Trivial disjunction
            return True
        elif (self in label) or (other in label):
            return True
        else:
            return False

    def And(self, other, label=None):
        if type(other) is not LTL_plus:
            raise ValueError('Type of {} must be LTL_plus.'
                             .format(other))
        if ('not_' in self) and (self in label):
            return False
        elif ('not_' in other) and (other in label):
            return False
        elif (self in label) and (other in label):
            return True
        elif (self in label) and ('not_' in other) and (other not in label):
            return True
        elif (other in label) and ('not_' in self) and (self not in label):
            return True
        else:
            return False

    def split(self):
        """
        @brief return a list of LTL_plus parts instead of type(str).
        """
        return [LTL_plus(part) for part in  str.split(self)]

    def eval(self, label):
        """
        @brief Gien a label from a TS, evaluate if the proposition is true.

        Only works (currently) with one connector.
        """
        if self == LTL_plus.empty or label == LTL_plus.empty:
            return True
        num_supported_connectors = 1
        num_supported_parts = (2*num_supported_connectors + 1)
        prop_parts = self.split()
        num_invalid_parts = len(prop_parts) - num_supported_parts
        if num_invalid_parts > 0:
            num_invalid_parts = len
            return ValueError("LTL_plus.eval only supports {} connectors. {} "
                              "were supplied".format((num_supported_connectors,
                                                      num_invalid_parts)))
        if num_invalid_parts == 0:
            for connector_num  in range(num_supported_connectors):
                connect_idx = 2*connector_num+1
                connector = prop_parts[connect_idx]
                if connector.lower() == 'and':
                    return prop_parts[connect_idx-1].And(
                                                     prop_parts[connect_idx+1],
                                                     label)
                elif connector.lower() == 'or':
                    return prop_parts[connect_idx-1].Or(
                                                     prop_parts[connect_idx+1],
                                                     label)
                else:
                    raise ValueError("Unsupported connector: {}".format(connector))
        if num_invalid_parts == -2:
            # Check to see if the single proposition satisfies the label.
            label_parts = label.split()
            if (len(label_parts) is 1) and (len(prop_parts) is 1):
                if label_parts[0] == prop_parts[0]:
                    return True
                elif ~label_parts[0] == prop_parts[0]:
                    return False
                elif ('not_' in label_parts[0]) or ('not_' in prop_parts[0]):
                    #E.g., 'red' satisfies 'not_yellow '
                    return True
                else:
                    # E.g., 'red' does not satisfy 'yellow'
                    return False
            if len(label_parts) == num_supported_parts:
                # E.g., swap to check if label="red and not_yellow" satisfies
                # self="red"
                return label.eval(self) #swapped recursion.

class DFA(object):
    """This is a deterministic Finite State Automaton (DFA).
    """
    def __init__(self, initial_state=None, alphabet=None,
                 transitions=dict([]), final_states=None, memory=None):
        self.state_transitions = {}
        if final_states is None:
            self.final_states = []
        else:
            self.final_states = final_states
        self.state_transitions=transitions
        if alphabet == None:
            self.alphabet=[]
        else:
            self.alphabet=alphabet
        self.initial_state = initial_state
        self.current_state = self.initial_state
        self.states =[ initial_state ] # the list of states in the machine.

    def __del__(self):
        """
        @brief Destroy the DFA
        """

    def reset (self):

        """This sets the current_state to the initial_state and sets
        input_symbol to None. The initial state was set by the constructor
         __init__(). """

        self.current_state = self.initial_state
        self.input_symbol = None

    def add_transition(self,input_symbol, state, next_state=None):
        if next_state is None:
            next_state = state
        else:
            self.state_transitions[(input_symbol, state)] = next_state
        if next_state in self.states:
            pass
        else:
            self.states.append(next_state)
        if state in self.states:
            pass
        else:
            self.states.append(state)
        if input_symbol in self.alphabet:
            pass
        else:
            self.alphabet.append(input_symbol)

    def get_transition(self, input_symbol, state):
        """This returns a list of next states given an input_symbol and state.
        """
        if (input_symbol, state) in self.state_transitions:
            return self.state_transitions[(input_symbol, state)]
        else:
            return None

    def predecessor(self,s):
        """
        list a set of predecessor for state s.
        """
        transFrom=set([])
        for a in self.alphabet:
            if (a,s) in self.state_transitions:
                transFrom.add((s,a, self.get_transition(a,s)))
        return transFrom

    def accessible(self):
        """
        list a set of reachable states from the initial state.
        Used for pruning.
        """
        reachable, index = [self.initial_state], 0
        while index < len(reachable):
            state, index = reachable[index], index + 1
            for (s0, a, s1) in self.transitionsFrom(state):
                if s1 not in reachable:
                    reachable.append(s1)
        states = []
        for s in reachable:
            states.append(s)
        transitions = dict([])
        for ((a,s0),s1) in self.state_transitions.items():
            if s0 in states and s1 in states:
                transitions[(a,s0)] = s1
        self.states=states
        self.state_transitions = transitions
        return

    def Trim(self):
        # remove states that are not reachable from the initial state.
        reachable, index = [self.initial_state], 0
        while index < len(reachable):
            state, index = reachable[index], index + 1
            for (s0, a, s1) in self.transitionsFrom(state):
                if s1 not in reachable:
                    reachable.append(s1)
        endable, index = list(self.final_states), 0
        while index < len(endable):
            state, index = endable[index], index + 1
            for ((a,s0),s1) in self.state_transitions.items():
                if s1 == state and s0 not in endable:
                    endable.append(s0)
        states = []
        for s in reachable:
            if s in endable:
                states.append(s)
        if not states:
            print("NO states after trimming. Null FSA.")
            return None
        transitions = dict([])
        for ((a,s0),s1) in self.state_transitions.items():
            if s0 in states and s1 in states:
                transitions[(a,s0)] = s1
        self.states=states
        self.state_transitions = transitions
        return

    def toDot(self,filename):
        f = open(filename,'w')
        f.write("digraph G { rankdir = LR\n")
        i = 0
        indexed_states = {}
        for state in self.states:
            indexed_states[state]=i
            if state in self.final_states:
                f.write("\t"+str(i) + "[label=\"" + str(state)
                        + "\",shape=doublecircle]\n")
            else:
                f.write("\t"+str(i) + "[label=\"" + str(state)
                        + "\",shape=circle]\n")
            i += 1
        for ((a,s0),s1) in self.state_transitions.items():
            f.write("\t"+str(indexed_states[s0])
                    + "->"
                    + str(indexed_states[s1])
                    + "[label = \""
                    + str(a)
                    + "\"]\n")
        f.write("}")
        f.close()


class DRA(DFA):
    """A child class of DFA --- determinisitic Rabin automaton
    """
    def __init__(self, initial_state=None, alphabet=None, transitions=dict([]),
                 rabin_acc= None, memory=None):
        # The rabin_acc is a list of rabin pairs
        # rabin_acc=[(J_i, K_i), i =0,...,N]
        # Each K_i, J_i is a set of states.
        # J_i is visited only finitely often
        # K_i has to be visited infinitely often.
        super(DRA, self).__init__(initial_state, alphabet, transitions)
        self.acc=rabin_acc

    def add_rabin_acc(self, rabin_acc):
        self.acc=rabin_acc


if __name__=='__main__':
    #construct a DRA, which is a complete automaton.
    # we use 'E' to stand for everything else other than 1,2,3,4.
    dra=DRA(0,['1','2','3','4','E'])
    dra.add_transition('2',0,0)
    dra.add_transition('3',0,0)
    dra.add_transition('E',0,0)

    dra.add_transition('1',0,1)
    dra.add_transition('1',1,2)
    dra.add_transition('3',1,2)
    dra.add_transition('E',1,2)

    dra.add_transition('2',1,3)

    dra.add_transition('1',2,2)
    dra.add_transition('3',2,2)
    dra.add_transition('E',2,2)

    dra.add_transition('2',2,3)

    dra.add_transition('1',3,3)
    dra.add_transition('2',3,3)
    dra.add_transition('E',3,3)

    dra.add_transition('3',3,0)

    dra.add_transition('4',0,4)
    dra.add_transition('4',1,4)
    dra.add_transition('4',2,4)
    dra.add_transition('4',3,4)
    dra.add_transition('4',4,4)
    dra.add_transition('1',4,4)
    dra.add_transition('2',4,4)
    dra.add_transition('3',4,4)
    dra.add_transition('E',4,4)
    dra.toDot('test_diagram')

    J0={4}
    K0={1}
    rabin_acc=[(J0,K0)]
    dra.add_rabin_acc(rabin_acc)
    # Define complemente with ~ operator override.



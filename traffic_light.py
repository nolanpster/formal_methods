#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

from pprint import pprint

from TS import TS
from DFA_TS_product import DFATimesTS
from NFA_DFA_Module.DFA import DFA
from NFA_DFA_Module.DFA import LTL_plus

class trafficLight(TS):
    """
    @breif A transition system of a traffic light.
    """
    # Class variables.
    state_variables = {'green', 'yellow', 'red'}
    actions = {'change'}
    log_file = "output_traffic_light.txt"

    def __init__(self, additional_state_variables=set([]), actions=None,
                 state_transitions=None, initial_state=None,
                 atom_prop=None):
        """
        @brief Construct a traffic light.
        """
        state_variables = (trafficLight.state_variables
                           | additional_state_variables)
        super(trafficLight, self).__init__(state_variables, actions,
                                           state_transitions,  initial_state,
                                           atom_prop)

    def __del__(self):
        """
        @brief Destroy the traffic light.
        """
        print 'Bzzt, trafficLight deleted'



if __name__=='__main__':
    # Construct the Transistion system of the traffic light.
    atom_prop = ['yellow',  'red']
    actions = ['change','E'] # 'E' is empty action
    initial_state = 'green'
    tl_1 = trafficLight(atom_prop=atom_prop, actions=actions,
                        initial_state=initial_state)
    tl_1.add_transition('E','green')
    tl_1.add_transition('change','green','yellow')
    tl_1.add_transition('E','yellow')
    tl_1.add_transition('change','yellow','red')
    tl_1.add_transition('E','red')
    tl_1.add_transition('change','red', 'redyellow')
    tl_1.add_transition('E', 'redyellow')
    tl_1.add_transition('change', 'redyellow','green')
    #tl_1.doAction('change')
    tl_1.toDot('trafficLight_1.dot')
    # Note, after X changes!
    with open(trafficLight.log_file,"w+") as log_file:
        pprint("The Transistion system of the traffic light:", log_file)
        pprint('', log_file)
        pprint(vars(tl_1), log_file)
        pprint('', log_file)
        pprint('', log_file)
    # Note how extra states are removed.
    tl_1.accessible()
    # Relabel after pruning this is used in TS*DFA formation.
    tl_1.labelStates()
    with open(trafficLight.log_file,"w+") as log_file:
        pprint("The pruned Transistion system of the traffic light:", log_file)
        pprint('', log_file)
        pprint(vars(tl_1), log_file)
        pprint('', log_file)
        pprint('', log_file)
    # Construct a DFA, to represent the safety property "all reds must be
    # immediately preceded by a yellow" in the traffic light TS. See figures
    # 4.10 and 4.11 in Principals of Model Checking for this case. We use 'E'
    # to stand for everything else other than specified inputs from the
    # alphabet.
    red = LTL_plus('red')
    yellow = LTL_plus('yellow')
    empty = LTL_plus('E')
    safety_spec_dfa=DFA(initial_state='q0', alphabet=[red, yellow],
                        final_states=['q0','q1'])
    # Specify Transitions
    safety_spec_dfa.add_transition(~yellow + 'Or' + ~red, 'q0', 'q0')
    safety_spec_dfa.add_transition(~yellow, 'q1', 'q0')
    safety_spec_dfa.add_transition(yellow, 'q1', 'q1')
    safety_spec_dfa.add_transition(~red + 'And' + yellow, 'q0', 'q1')
    safety_spec_dfa.add_transition(~red + 'And' + ~yellow, 'q0', 'q0')
    # The state implying a bad-prefix.
    safety_spec_dfa.add_transition(red, 'q0', 'q2')
    safety_spec_dfa.add_transition(empty, 'q2', 'q2')
    safety_spec_dfa.add_transition(empty, 'q0', 'q0')
    safety_spec_dfa.add_transition(empty, 'q1', 'q0') # E |= ~yellow
    safety_spec_dfa.toDot('dfaSafetySpec.dot')

    # The complement of the dfa satisfies the negation of the specification.
    bad_prefi_dfa = ~safety_spec_dfa
    bad_prefi_dfa.toDot('dfaBadPref.dot')
    with open(trafficLight.log_file,"a") as log_file:
        pprint("The DFA representing the negation of the safety property "
               "\"every {red} must be preceded by a {yellow}:\"", log_file)
        pprint('', log_file)
        pprint(vars(bad_prefi_dfa), log_file)
        pprint('', log_file)
        pprint('', log_file)
    bad_prefi_dfa.Trim()
    safety_spec_dfa.Trim()
    # Construct the product of the TS and the DFA-complement.
    verify_tl_safe = DFATimesTS(tl_1, bad_prefi_dfa)
    with open(trafficLight.log_file,"a") as log_file:
        pprint("The product of a TS and DFA, that represents the negation of "
               "the safety property "
               "\"every {red} must be preceded by a {yellow}:\"", log_file)
        pprint('', log_file)
        pprint(vars(verify_tl_safe), log_file)
        pprint('', log_file)
        pprint('', log_file)
    # After two actions we are at the augmented statee = (('red',), 'q0')
    verify_tl_safe.delta('change')
    verify_tl_safe.delta('change')
    with open(trafficLight.log_file,"a") as log_file:
        pprint("After two \"change\" actions, the product system is at the "
               " following state:", log_file)
        pprint('', log_file)
        pprint(verify_tl_safe.current_state, log_file)
        pprint('', log_file)
        pprint('', log_file)
    # The third action:
    verify_tl_safe.delta('change')
    with open(trafficLight.log_file,"a") as log_file:
        pprint("After three \"change\" actions, the product system is at the "
               " following state:", log_file)
        pprint('', log_file)
        pprint(verify_tl_safe.current_state, log_file)
        pprint('', log_file)
    # The fourth action:
    verify_tl_safe.delta('change')
    with open(trafficLight.log_file,"a") as log_file:
        pprint("After three \"change\" actions, the product system is at the "
               " following state:", log_file)
        pprint('', log_file)
        pprint(verify_tl_safe.current_state, log_file)
        pprint('', log_file)
        pprint('', log_file)

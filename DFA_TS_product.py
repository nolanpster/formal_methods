#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

import  tulip.transys.mathset as MS

from NFA_DFA_Module.DFA import DFA
from NFA_DFA_Module.DFA import LTL_plus
from TS import TS


class DFATimesTS(TS):
    """
    @brief Product of a Transition System, without terminal states, and a
           Deterministic Finite Automata.

           Can be used for checking safety properties of a TS.
    """
    def __init__(self, trans_sys, df_automaton):
         """
         @brief comput the product of a @ref TS and @ref DFA.
         """
         self.trans_sys = trans_sys # State of TS is actually updated upon action.
         self.df_automaton = df_automaton # Used for initialization. States not updated.
         self.update()
         # Set @c current_state with @ref delta and then update initial state.
         self.delta((trans_sys.initial_state, df_automaton.initial_state))

    def update(self):
        # Used if augmenting a TS or DFA on the fly
        # Only 'add' states, no deletions!
         S = MS.MathSet(self.trans_sys.states)
         Q = MS.MathSet(self.df_automaton.states)
         self.states = S*Q #MathSet implements cartesian product.
         self.state_variables = self.states # States form a set already.
         self.actions = self.trans_sys.actions
         self.atom_prop = tuple(TS.str2Tup(s) for s in self.df_automaton.states)
         self.labelStates()
         self.addTransitionsFrom(self.trans_sys, self.df_automaton)
         # Keep current state.


    def delta(self, aug_state=None, action=None):
         """
         @brief evaluate the label of the input @aug_state.

         @param aug_state a set {TS.state, DFA.state}
         @param action Provide LTL_plus.empty for empty action. If action
                is not provided, this sets the intial state.
         """
         if action is not None and aug_state is not None:
             # Process action from state. ...
             # Don't think this input set would ever actually be provided.
             self.current_state = self.get_transition(action, aug_state)
             self.TS.doAction(action)
         elif action is not None and aug_state is None:
             # Process action on TS system.
             self.current_state = self.get_transition(action,
                                                      self.current_state)
         else:
             # Assume we're setting the initial state.
             dfa_q0 = aug_state[1]
             ts_s0 = aug_state[0]
             ts_s0_label =  tuple(TS.evalProposition(self.trans_sys.atom_prop,
                                  ts_s0))
             if not ts_s0_label:
                 # Correct to 'empty' label, it's bad style though.
                 ts_s0_label = LTL_plus.empty
             dfa_q_init = self.df_automaton.get_transition(ts_s0_label[0],
                                                           dfa_q0)
             self.current_state = (ts_s0, dfa_q_init)
             self.initial_state = self.current_state

    def addTransitionsFrom(self, trans_sys, df_automaton):
        """
        @brief iterate through the TS's state action pairs and,
        given the label, identify the next state in the DFA.
        """
        self.state_transitions = {}
        for act_state_pair, ts_next_state in trans_sys.state_transitions.items():
            ts_action = act_state_pair[0]
            ts_state = act_state_pair[1]
            ts_next_label = tuple(trans_sys.labeled_states[ts_next_state])
            empty_action = False
            if ts_next_label not in trans_sys.atom_prop:
                empty_action = True
            elif ts_next_label == ():
                empty_action = True
            for dfa_state in df_automaton.states:
                if empty_action:
                    #dfa_next_state = dfa_state
                    ts_next_label = LTL_plus.empty
                #else:
                input_symbol = LTL_plus(ts_next_label[0])
                dfa_next_state = df_automaton.get_transition(input_symbol,
                                                                 dfa_state)
                if dfa_next_state is None:
                    # check to see if the current input_symbol satisfies any
                    # any actions at other states.
                    satisfying_symbol = df_automaton.findEquivalentTransition(
                                                                  input_symbol,
                                                                  dfa_state)
                    dfa_next_state = df_automaton.get_transition(
                                                             satisfying_symbol,
                                                             dfa_state)
                    if dfa_next_state is None:
                        break
                self.add_transition(action=ts_action,
                                    state=tuple((ts_state, dfa_state)),
                                    next_state=tuple((ts_next_state,
                                                      dfa_next_state)))

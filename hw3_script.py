#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

from pprint import pprint

from TS import TS
from DFA_TS_product import DFATimesTS
from NFA_DFA_Module.DFA import DFA
from NFA_DFA_Module.DFA import LTL_plus
from MDP import MDP


if __name__=='__main__':
    prob_dict = {'a': {
                         ('1', '1'): 0.7,
                         ('1', '3'): 0.3,
                         ('2', '1'): 0.3,
                         ('2', '3'): 0.7,
                         ('3', '3'): 1
                         },
                 'b': {
                         ('1', '2'): 0.6,
                         ('1', '3'): 0.4,
                         ('2', '2'): 0.5,
                         ('2', '3'): 0.5,
                         ('3', '3'): 1
                         }
                 }
    import pdb; pdb.set_trace()
    quiz_mdp = MDP(init='1', actList=['a','b'], states=['1', '2', '3'],
                   prob=prob_dict, gamma=1)
    quiz_mdp.solve() # Defaults to value iteration algorithm.

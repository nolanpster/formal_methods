#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

from TS import TS

class trafficLight(TS):
    """
    @breif A transition system of a traffic light.
    """
    # Class variables.
    state_variables = {'green', 'yellow', 'red'}
    actions = {'change'}

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
    tl_1.add_transition('change','red','redyellow')
    tl_1.add_transition('E','redyellow')
    tl_1.add_transition('change','redyellow','green')
    tl_1.doAction('change')
    tl_1.toDot('trafficLight_1.dot')



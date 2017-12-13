#!/usr/bin/env python

import astar
from NFA_DFA_Module.DFA import LTL_plus
import numpy as np


class GridGraph(object):


    # Class variable - graph connectivity.
    connect4 = 4
    # [East, South, West, North]
    acts = ['East', 'South', 'West', 'North']
    # Motion primitive: row 0 is dX, row 1 is dy.
    motion_prim = np.array([[1, 0, -1, 0],
                            [0, 1, 0, -1]], dtype=np.int8)
    # Label of obstacle
    red = LTL_plus('red')

    def __init__(self, paths={}, grid_map=None, neighbor_dict=None, label_dict=None):
        # Shortest Path dictionary structure = {{s_0, s_N}: (s_0, ... s_N)} Note these are hard-coded chosing arbitrary
        # paths when there are multiple.  Note, current implementation just reverses the sequence for going the reverse
        # direction, there are probably a million ways to make this whole structure more efficient; I can think of 3
        # right now ...
        self.paths = paths
        self.grid_map = grid_map
        self.astar_map = np.zeros(self.grid_map.shape, dtype=np.int8)
        # Fill in astar_map with ones for every obstacle.
        if label_dict is not None:
            for state, label in label_dict.iteritems():
                if label==self.red:
                    grid_row, grid_col = np.where(self.grid_map==int(state))
                    self.astar_map[grid_row, grid_col] = 1
        self.neighbor_dict = neighbor_dict
        if self.grid_map is not None and self.neighbor_dict is not None:
            self.setStateTransitionsFromActions()
            self.m, self.n = self.grid_map.shape
        else:
            self.state_transition_actions = {}

    def getShortestPath(self, s_0, s_N):
        # Return a path from a starting state, s_0, to a final state, s_N.
        #
        # @todo Can save _partial_ paths too!

        # Test if key is in dictionary
        if (s_0, s_N) not in self.paths:
            yA, xA = np.where(self.grid_map==s_0)
            yB, xB = np.where(self.grid_map==s_N)
            act_idx = astar.pathFind(self.astar_map, self.n, self.m, self.connect4, self.motion_prim[0],
                                     self.motion_prim[1], xA[0], yA[0], xB[0], yB[0])
            if act_idx:
                self.paths[(s_0, s_N)] = self.actionListToStateNum(yA[0], xA[0], act_idx)
            else:
                self.paths[(s_0, s_N)] = None
        return self.paths[(s_0, s_N)]

    def actionListToStateNum(self, s_0_row, s_0_col, action_idx):
        """
        @brief Convert actions taken from state-0 to list of states.
        """
        path = [self.grid_map[s_0_row, s_0_col]]
        map_col_row = np.array([s_0_col, s_0_row], dtype=np.int8)
        for act_idx in action_idx:
            map_col_row += self.motion_prim[:,int(act_idx)]
            path.append(self.grid_map[map_col_row[1], map_col_row[0]])
        return path

    def shortestPathLength(self, s_0, s_N):
        path = self.getShortestPath(s_0, s_N)
        return len(path) if path is not None else 0

    def setStateTransitionsFromActions(self):
        """
        @brief Placeholder method for an automated way to do this
        """
        self.state_transition_actions = {}
        for start_state in range(self.grid_map.size):
            for neighbor_state, action in self.neighbor_dict[str(start_state)].iteritems():
                self.state_transition_actions[(start_state, neighbor_state)] = action

    def getObservedAction(self, s_0, s_N):
        """
        @brief dummy implementation. Should not be using hard-coded state-action pairs.
        """
        # Test if _key_ is in dictionary.
        if (s_0, s_N) in self.state_transition_actions:
            return self.state_transition_actions[(s_0, s_N)]
        else:
            return None

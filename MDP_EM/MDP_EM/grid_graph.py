#!/usr/bin/env python

import astar
from NFA_DFA_Module.DFA import LTL_plus
import numpy as np


class GridGraph(object):


    # Class variables
    # graph connectivity.
    connect4 = 4
    # [East, South, West, North]
    acts = ['East', 'South', 'West', 'North']
    # Motion primitive: row 0 is dX, row 1 is dy.
    motion_prim = np.array([[1, 0, -1, 0],
                            [0, 1, 0, -1]], dtype=np.int8)
    # Label of obstacle
    red = LTL_plus('red')
    OBSTACLE_VAL = 1

    def __init__(self, paths={}, distances={}, grid_map=None, neighbor_dict=None, label_dict=None,
                 obstacle_label=red, state_idx_to_observe=0):
        # Shortest Path dictionary structure = {{s_0, s_N}: (s_0, ... s_N)}
        self.paths = paths
        self.distances = distances
        self.grid_map = grid_map
        self.astar_map = np.zeros(self.grid_map.shape, dtype=np.int8)
        self.astar_no_obstacle_map = np.zeros(self.grid_map.shape, dtype=np.int8)
        self.obstacle_label = obstacle_label
        self.state_idx_to_observe = state_idx_to_observe

        # The longest path length is returned if no path is found (the state is inside an obstacle). Since these values
        # are used in the gaussian kernel functions, exp(-746) is the first value that returns 0.0, which is desired for
        # a state inside an obstacle; we want that state not to increase the  weight of phi.
        self.longest_path_length_squared = min(746**2, self.grid_map.size**2)
        # Fill in astar_map with ones for every obstacle.
        if label_dict is not None:
            for state, label in label_dict.iteritems():
                if label==self.obstacle_label:
                    grid_row, grid_col = np.where(self.grid_map==state[self.state_idx_to_observe])
                    self.astar_map[grid_row, grid_col] = GridGraph.OBSTACLE_VAL
        self.neighbor_dict = neighbor_dict
        if self.grid_map is not None and self.neighbor_dict is not None:
            self.setStateTransitionsFromActions()
            self.m, self.n = self.grid_map.shape
        else:
            self.state_transition_actions = {}

    def getShortestPath(self, s_0, s_N):
        # Return a path from a starting state, s_0, to a final state, s_N. NOTE! As a huristic if s_0 is in an
        # obstacle, then the path is set to np.inf. If s_N is in an obstacle, then paths are stil valid (i.e., it is
        # still possible for a policy to be dependent on the position of an obstacle).
        #
        # @todo Can save _partial_ paths too!
        # @todo There's a more efficient way of finding coordinates in a known grid than np.where().

        # Test if key is in dictionary
        if (s_0, s_N) not in self.paths:
            if s_0 == s_N:
                self.paths[(s_0, s_N)] = [s_N]
            else:
                yA, xA = np.where(self.grid_map==s_0)
                yB, xB = np.where(self.grid_map==s_N)
                if self.astar_map[yB, xB] == GridGraph.OBSTACLE_VAL:
                    # Goal state is in obstacle.
                    act_idx = astar.pathFind(self.astar_no_obstacle_map, self.n, self.m, self.connect4,
                                             self.motion_prim[0], self.motion_prim[1], xA[0], yA[0], xB[0], yB[0])
                    self.paths[(s_0, s_N)] = self.actionListToStateNum(yA[0], xA[0], act_idx)
                elif self.astar_map[yA, xA] == GridGraph.OBSTACLE_VAL:
                    # Start state is in obstacle.
                    self.paths[(s_0, s_N)] = None
                else:
                    act_idx = astar.pathFind(self.astar_map, self.n, self.m, self.connect4, self.motion_prim[0],
                                             self.motion_prim[1], xA[0], yA[0], xB[0], yB[0])
                    self.paths[(s_0, s_N)] = self.actionListToStateNum(yA[0], xA[0], act_idx)
        return self.paths[(s_0, s_N)]

    def getEuclidianDistance(self, s_0, s_N):
        """
        @brief Return the Euclidan distance between two cells.

        @todo There's a more efficient way of finding coordinates in a known grid than np.where().
        """
        raise NotImplementedError('Need to figure out if obstacles effect distance calculation at all.')
        if (s_0, s_N) not in self.distances:
            yA, xA = np.where(self.grid_map==s_0)
            yB, xB = np.where(self.grid_map==s_N)

            del_x = x_B - x_A
            del_y = y_B - y_A
            self.distances[(s_0, s_N)] = np.sqrt(del_x**2 + del_y**y)

        return self.distances[(s_0, s_N)]

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
        """
        @brief Returns the length of the shortest path from s_0 to s_N. If states are one move apart, the path includes
        the start and end state, (length=2) so we must subtract one from it for the correct return value.
        """
        path = self.getShortestPath(s_0, s_N)
        return len(path)-1 if path is not None else self.longest_path_length_squared

    def setStateTransitionsFromActions(self):
        """
        @brief Creates a dictionary of actions between neighboring cells.
        """
        self.state_transition_actions = {}
        for start_state in range(self.grid_map.size):
            for neighbor_state, action_idx in self.neighbor_dict[start_state].iteritems():
                self.state_transition_actions[(start_state, neighbor_state)] = action_idx

    def getObservedAction(self, s_0, s_N):
        """
        @brief Returns the action index taken to go between two neighboring states from s_0 to s_N.
        """
        # Test if _key_ is in dictionary.
        if (s_0[0][self.state_idx_to_observe], s_N[0][self.state_idx_to_observe]) in self.state_transition_actions:
            return self.state_transition_actions[(s_0[0][self.state_idx_to_observe], s_N[0][self.state_idx_to_observe])]
        else:
            return None

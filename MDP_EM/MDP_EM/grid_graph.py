#!/usr/bin/env python

class GridGraph(object):
    # Doesnt' really look like a graph yet.

    def __init__(self, paths, grid_map):
        # Shortest Path dictionary structure = {{s_0, s_N}: (s_0, ... s_N)} Note these are hard-coded chosing arbitrary
        # paths when there are multiple.  Note, current implementation just reverses the sequence for going the reverse
        # direction, there are probably a million ways to make this whole structure more efficient; I can think of 3
        # right now ...
        self.paths = paths
        self.grid_map = grid_map

    def getShortestPath(self, s_0, s_N):
        # Return a path from a starting state, s_0, to a final state, s_N.
        # @NOTE the path might be the wrong direction, but at this point only @ref shortestPathLength uses the path for 
        # the path lenght.
        path = self.paths.get(frozenset((s_0, s_N)))
        return path

    def shortestPathLength(self, s_0, s_N):
        path = self.getShortestPath(s_0, s_N)
        return len(path) if path is not None else 0

    def setStateTransitionsFromActions(self, neighbor_dict):
        """
        @brief Placeholder method for an automated way to do this
        """
        self.state_transition_actions = {}
        for start_state in range(self.grid_map.size):
            for neighbor_state, action in neighbor_dict[str(start_state)].iteritems():
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

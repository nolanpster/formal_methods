#!/usr/bin/env python

class GridGraph(object):
    # Doesnt' really look like a graph yet.

    def __init__(self, paths):
        # Shortest Path dictionary structure = {{s_0, s_N}: (s_0, ... s_N)} Note
        # these are hard-coded chosing arbitrary paths when there are multiple.
        # Note, current implementation just reverses the sequence for going the
        # reverse direction, there are probably a million ways to make this whole
        # structure more efficient; I can think of 3 right now ...
        self.paths = paths

    def getShortestPath(self, s_0, s_N):
        # Return a path from a starting state, s_0, to a final state, s_N.
        path = self.paths.get(frozenset((s_0, s_N)))
        return path

    def shortestPathLength(self, s_0, s_N):
        path = self.getShortestPath(s_0, s_N)
        return len(path) if path is not None else 0

    def setStateTransitionsFromActions(self, state_transition_actions):
        """
        @brief Placeholder method for an automated way to do this.abs
        """
        self.state_transition_actions = state_transition_actions

    def getObservedAction(self, s_0, s_N):
        """
        @brief dummy implementation. Should not be using hard-coded state-action pairs.
        """
        return self.state_transition_actions[(s_0, s_N)]


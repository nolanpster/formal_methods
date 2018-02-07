#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu, Nishant Doshi, ndoshi@wpi.edu'


from abc import ABCMeta
from abc import abstractmethod

class KernelCenter(object):
    __metaclass__ = ABCMeta

    def __init__(self, center, graph):
        self.center = center
        self.graph = graph

    @abstractmethod
    def __call__(self, state):
        """
        @brief This method must be overloaded.
        """
        raise NotImplementedError


class GeodesicGaussianKernelCenter(KernelCenter):
    """
    @brief Geodesic Gausian Kernels from Sugiyama Ch. 3.
    """

    def __init__(self, center, graph):
        """
        """
        super(self.__class__, self).__init__(center, graph)


    def __call__(self, state):
        return self.graph.shortestPathLength(state, self.center)


class OrdinaryGaussianKernelCenter(KernelCenter):
    """
    @brief Ordinary Gausian Kernels from Sugiyama Ch. 3.
    """

    def __init__(self, center, graph):
        """
        """
        super(self.__class__, self).__init__(center, graph)
        raise NotImplementedError

    def __call__(self, state):
        return self.graph.getEuclidianDistance(state, self. center)


# -*- coding: utf-8 -*-
"""
:mod:`metaopt.algo.evol` -- Perform evolutionary optimization on hyperparameters 
==================================================================================

.. module:: evol
   :platform: Unix
   :synopsis: Use Evolutionary processes 

"""
import numpy

from metaopt.algo.base import BaseAlgorithm


class EvolutionaryOptimizer(BaseAlgorithm):
    """Wrapper skopt's bayesian optimizer"""

    def __init__(self, space, **kwargs):
        super(EvolutionaryOptimizer, self).__init__(space)

        self.optimizer = Optimizer(
            base_estimator=EvolutionaryProcess(**kwargs),
            dimensions=dimension)

        self.strategy = "cl_min"

    def suggest(self, num=1):
        """Suggest a `num`ber of new sets of parameters.

        Perform a step towards negative gradient and suggest that point.

        """
        points = self.optimizer.ask(n_points=num, strategy=self.strategy)
        return points

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        Save current point and gradient corresponding to this point.

        """
        self.optimizer.tell(points, [r['objective'] for r in results])

    @property
    def is_done(self):
        """Implement a terminating condition."""
        return False

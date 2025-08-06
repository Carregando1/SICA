#State: Attempt 2 version.

from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray

from cax_sica.genetic.transition import transition


@dataclass
class CurrentState:
    """
    Dataclass that implements the current state of the genetic algorithm.

    the shape of 'initial' is (width, length), with all entries being integers
    between 0 and states-1.

    the shape of 'rules' is (height-1, width, length, 18), where the
    18 is (neighbor-states)*(self-states).

    the shape of 'ruleindices' is either none or the same as 'rules'
    """

    initial: NDArray[np.int32]
    rules: NDArray[np.int32]
    ruleindices: NDArray[np.int32] | None = None
    _generated: NDArray[np.int32] | None = None

    def states(self):
        """
        Just returns 2 for number of states due to transition restrictions.
        """
        return 2

    # generate results get cached.
    def generate(self) -> NDArray[np.int32]:
        """
        Returns the effect of applying the spacetime-inhomogeneous set of rules.
        """
        if self._generated is not None:
            return self._generated
        grid = transition(self.initial, self.rules, self.rules.shape[0])
        self._generated = grid
        return grid

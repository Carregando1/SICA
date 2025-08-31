#State: Now merged with transition.

from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray

import jax
import jax.numpy as jnp
from flax import nnx
from cax_sica.cax_new.systems.sica import sica

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

    # generate results get cached.
    def generate(self) -> NDArray[np.int32]:
        """
        Returns the effect of applying the spacetime-inhomogeneous set of rules.
        """
        if self._generated is not None:
            return self._generated
        
        seed = 0
        num_steps = self.rules.shape[0]
        time = 0
        rngs = nnx.Rngs(seed)

        ca = sica(rngs=rngs)
        state_init = jnp.zeros((self.initial.shape[0]+2, self.initial.shape[1]+2, 1)).at[1:-1, 1:-1].set(self.initial.reshape(self.initial.shape[0], -1, 1))
        srt = jnp.zeros((self.rules.shape[0], self.rules.shape[1]+2, self.rules.shape[2]+2, self.rules.shape[3])).at[:, 1:-1, 1:-1, :].set(self.rules)
        ca.update.update_srt(srt=srt)
        assert num_steps <= srt.shape[0], f"Requested time {num_steps} is out of bounds, max time {srt.shape[0]}"
        state_final, states = ca(state=state_init, time=time, num_steps=num_steps)
        states = jnp.concatenate([(state_init[1:-1, 1:-1].astype(int)).reshape(1, self.initial.shape[0], -1), (states[:, 1:-1, 1:-1].astype(int)).reshape(states.shape[0], self.initial.shape[0], -1)])
        
        self._generated = states
        return states

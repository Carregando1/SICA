
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from cax_sica.cax_new.core.update.update import Update
from cax_sica.cax_new.types import Input, Perception, Rule, State


class sicaUpdate(Update):

	def __init__(self, rngs: nnx.Rngs):
		#first 3 dimensions (time * width * height) are modifiable; last one (18 rules) is not
		self.srt = np.zeros((100, 64, 64, 18))

	def __call__(self, state: State, time: int, perception: Perception, input: Input | None = None) -> State:
		#No assertion here as it raises an error
		self_alive = perception[..., 0:1]
		num_alive_neighbors = perception[..., 1:2].astype(jnp.int32)
		position = np.zeros(self_alive.shape[:-1]).astype(np.int32)
		position = np.concatenate((np.indices(self_alive.shape[:-1])[0].reshape(self_alive.shape[:-1]+(1,)), np.indices(self_alive.shape[:-1])[1].reshape(self_alive.shape[:-1]+(1,))), axis=2)
		#Scan SRT for each position
		state = jnp.where(self.srt[time, position[:,:,0], position[:,:,1], (9*self_alive.reshape(self_alive.shape[0], -1)[position[:,:,0], position[:,:,1]]+num_alive_neighbors.reshape(num_alive_neighbors.shape[0], -1)[position[:,:,0], position[:,:,1]]).astype(jnp.int32)] == 1, 1.0, 0.0)
		return state.reshape(state.shape[0], -1, 1)

	@nnx.jit
	def update_srt(self, srt) -> None:
		"""Update the SRT: takes in jnp arrays."""
		assert len(srt.shape) == 4, "SRTs must be 4-dimensional"
		assert srt.shape[3] == 18, "SRTs must have exactly 18 rules"
		self.srt = srt
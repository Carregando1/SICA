"""Model for 2-state Spacetime-Inhomogeneous Cellular Automata."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np #Fix
from flax import nnx
from jax import Array
import cax

from cax.core.perceive import ConvPerceive, identity_kernel, neighbors_kernel

from cax.core.update.update import Update
from cax.types import Input, Perception, Rule, State

from cax.core.ca import CA, metrics_fn
from cax.utils import clip_and_uint8

import jax.numpy as jnp
from flax import nnx

class SICAPerceive(ConvPerceive):
	"""2-State SICA Perceive class."""

	def __init__(self, rngs: nnx.Rngs, *, padding: str = "CIRCULAR"):
		"""Initialize SICAPerceive."""
		channel_size = 1
		super().__init__(
			channel_size=channel_size,
			perception_size=2 * channel_size,
			rngs=rngs,
			kernel_size=(3, 3),
			padding=padding,
			feature_group_count=channel_size,
		)

		kernel = jnp.concatenate([identity_kernel(2), neighbors_kernel(2)], axis=-1)
		kernel = jnp.expand_dims(kernel, axis=-2)
		self.conv.kernel = nnx.Param(kernel)

class SICAUpdate(Update):
	"""2-State SICA Update class."""

	def __init__(self, time, rngs: nnx.Rngs):
		"""Initialize SICAUpdate."""
		self.srt = np.zeros((1, 1, 1, 18))
		self.time = np.array([time])

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the SICA rules based on SRT input.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state, including cell state and neighbor count.
			input: Input to the cellular automaton (unused in this implementation).

		Returns:
			Updated state of the cellular automaton.

		"""
		self_alive = perception[..., 0:1]
		num_alive_neighbors = perception[..., 1:2].astype(jnp.int32)
		position = np.zeros(self_alive.shape[:-1]).astype(np.int32)
		position = np.concatenate((np.indices(self_alive.shape[:-1])[0].reshape(self_alive.shape[:-1]+(1,)), np.indices(self_alive.shape[:-1])[1].reshape(self_alive.shape[:-1]+(1,))), axis=2)
		state = jnp.where(self.srt[self.time[0], position[:,:,0], position[:,:,1], (9*self_alive.reshape(self_alive.shape[0], -1)[position[:,:,0], position[:,:,1]]+num_alive_neighbors.reshape(num_alive_neighbors.shape[0], -1)[position[:,:,0], position[:,:,1]]).astype(jnp.int32)] == 1, 1.0, 0.0)
		self.time += jnp.array([1])
		return state.reshape(state.shape[0], -1, 1)

	@nnx.jit
	def update_srt(self, srt) -> None:
		"""Update the SRT of a SICA from an array input.

		Args:
			srt: An NDArray with 4 dimensions and last dimension of size 18.

		"""
		assert len(srt.shape) == 4, f"Expected 4 dimensions in SRT, received {len(srt.shape)}"
		assert srt.shape[3] == 18, f"Expected 18 rules in SRT, received {srt.shape[3]}"
		self.srt = srt

class sica(CA):
	"""2-state Spacetime-Inhomogeneous Cellular Automata."""

	def __init__(self, time, rngs: nnx.Rngs, *, metrics_fn: Callable = metrics_fn):
		"""Initialize 2-state SICA."""
		perceive = SICAPerceive(rngs=rngs)
		update = SICAUpdate(time=time, rngs=rngs)
		super().__init__(perceive, update, metrics_fn=metrics_fn)

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB.

		Args:
			state: An array with two spatial/time dimensions.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		rgb = jnp.repeat(state, 3, axis=-1)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
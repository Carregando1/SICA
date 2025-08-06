
from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax_sica.cax_new.core.catime import CA, metrics_fn
from cax_sica.cax_new.types import State
from cax_sica.cax_new.utils import clip_and_uint8

from .sica_perceive import sicaPerceive
from .sica_update import sicaUpdate


class sica(CA):
	def __init__(self, rngs: nnx.Rngs, *, metrics_fn: Callable = metrics_fn):
		perceive = sicaPerceive(rngs=rngs)
		update = sicaUpdate(rngs=rngs)
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
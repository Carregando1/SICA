"""Model for 2-state Spacetime-Inhomogeneous Cellular Automata."""

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.perceive import ConvPerceive, identity_kernel, neighbors_kernel

from cax.core.update.update import Update
from cax.types import Input, Perception, Rule, State

from cax.core.ca import CA, metrics_fn
from cax.utils import clip_and_uint8

import jax.numpy as jnp
from flax import nnx

import math
import jax
jax.config.update("jax_enable_x64", True)

"""
!!! IMPORTANT !!!
!!! CHANGES IN SRT SYNTAX !!!
In this combinatoric generalization of SICA to n <= 12 states (n > 12 takes too much memory for a complete 
SRT allocation so is not allowed in this implementation), the SRT indexing routine has a significant 
difference to that of previous SICAProject versions, such as SICA_V2 or CAX_2S. The SRT indexing routine
implemented in this version is as follows:
- For a single cell *C*, let the state of *C* be C_state, and let the number of neighbors of *C* in each state 
[1, 2, ..., max_state] (note that this tuple excludes the state 0; this is intentional) be a tuple in the form 
(a_1, a_2, ..., a_n). Such a tuple will be notated as a "neighbor tuple". Note that a_1 + a_2 + ... + a_n 
must be equal to or less than 8 in order to satisfy the Moore neighborhood condition; 
the difference between 8 and this sum is equal to the number of neighbors in the state 0.
- We order all possible tuples satisfying a_1 + a_2 + ... + a_n ≤ 8 as follows:
  * All tuples are first sorted in descending value of a_1, so the tuple (8, 0, ..., 0) is first in the list
  (index 0);
  * All tuple groups with a_1 constant are then sorted in descending value of a_2;
  * All tuple groups with both a_1 and a_2 constant are then sorted in descending value of a_3, and so on.
  * Let the length of this list be order_len.
- Each SRT raw rule (C_state, a_1, a_2, ..., a_n) -> final_state is added to the SRT array as follows:
  * The index of the rule within the array is (C_state * order_len + the corresponding index of the neighbor 
  tuple (a_1, a_2, ..., a_n));
  * The value of the index is final_state.
- The neighbor tuple of the cell *C* is then matched to its corresponding index in the SRT to determine its 
final state.
For example, the ordering of the tuples with (n = max_state = 2, number of states = 3) is:
[(8,0), (7,1), (7,0), (6,2), (6,1), (6,0), (5,3), (5,2), (5,1), (5,0), (4,4), (4,3), (4,2), (4,1), (4,0), (3,5),
(3,4), (3,3), (3,2), (3,1), (3,0), (2,6), (2,5), (2,4), (2,3), (2,2), (2,1), (2,0), (1,7), (1,6), (1,5), (1,4),
(1,3), (1,2), (1,1), (1,0), (0,8), (0,7), (0,6), (0,5), (0,4), (0,3), (0,2), (0,1), (0,0)], with length 45.
A sample SRT ruleset with number of states = 3 (guaranteed to be of length 3 * 45 = 135) could start with:
[1,2,0,1,0,0,2,1,0,2,2,2,1,2,0,1,2,0, ...]
The final state of a sample cell with cell state 0 and neighbor tuple (a_1, a_2) = (4,2) would be determined by the
value of the SRT at index (0 * 45) + 12 (12 is the index of the tuple (4,2)). This value is 1.
Thus, the final state of this cell would be 1.
"""

class NSICAPerceive(ConvPerceive):
	"""2-State SICA Perceive class."""

	def __init__(self, rngs: nnx.Rngs, *, states: int = 2, padding: str = "SAME"):
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

		self.states = states
	
	def __call__(self, state: State) -> Perception:
		"""Apply perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The perceived state after applying convolutional layers.

		"""
		convolution = jnp.array(self.conv(10**state), dtype=jnp.int64)
		"""Convolutional workaround is constructed based on worst case of states=12 to avoid for-loops. Currently avoiding all for-loop routines. Should for-loops be used instead?"""
		alivestate = (
			jnp.array([jnp.round(jnp.log10(convolution[:,:,0]))]),
			jnp.array([convolution[:,:,1]%(10**1)]), 
			jnp.array([convolution[:,:,1]%(10**2)-convolution[:,:,1]%(10**1)])/(10**1),
			jnp.array([convolution[:,:,1]%(10**3)-convolution[:,:,1]%(10**2)])/(10**2),
			jnp.array([convolution[:,:,1]%(10**4)-convolution[:,:,1]%(10**3)])/(10**3),
			jnp.array([convolution[:,:,1]%(10**5)-convolution[:,:,1]%(10**4)])/(10**4),
			jnp.array([convolution[:,:,1]%(10**6)-convolution[:,:,1]%(10**5)])/(10**5),
			jnp.array([convolution[:,:,1]%(10**7)-convolution[:,:,1]%(10**6)])/(10**6),
			jnp.array([convolution[:,:,1]%(10**8)-convolution[:,:,1]%(10**7)])/(10**7),
			jnp.array([convolution[:,:,1]%(10**9)-convolution[:,:,1]%(10**8)])/(10**8),
			jnp.array([convolution[:,:,1]%(10**10)-convolution[:,:,1]%(10**9)])/(10**9),
			jnp.array([convolution[:,:,1]%(10**11)-convolution[:,:,1]%(10**10)])/(10**10),
			jnp.array([convolution[:,:,1]%(10**12)-convolution[:,:,1]%(10**11)])/(10**11),
		)
		return jnp.concatenate(alivestate, dtype=jnp.int32)

class NSICAUpdate(Update):
	"""2-State SICA Update class."""

	def __init__(self, time, states, rngs: nnx.Rngs):
		"""Initialize SICAUpdate."""
		self.srt = jnp.zeros((1, 1, 1, 18))
		self.time = jnp.array([time])
		self.states = states

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the SICA rules based on SRT input.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state, including cell state and neighbor count.
			input: Input to the cellular automaton (unused in this implementation).

		Returns:
			Updated state of the cellular automaton.

		"""
		self_state = perception[0]
		"""Currently avoiding all for-loop routines here. Should for-loops be used instead?"""
		indexcontributions = jnp.round(jnp.array([
			jax.scipy.special.factorial(6+self.states-perception[2])/jax.scipy.special.factorial(self.states-1)/jax.scipy.special.factorial(7-perception[2]),
			jax.scipy.special.factorial(5+self.states-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-2)/jax.scipy.special.factorial(7-perception[3]-perception[2]),
			jax.scipy.special.factorial(4+self.states-perception[4]-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-3)/jax.scipy.special.factorial(7-perception[4]-perception[3]-perception[2]),
			jax.scipy.special.factorial(3+self.states-perception[5]-perception[4]-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-4)/jax.scipy.special.factorial(7-perception[5]-perception[4]-perception[3]-perception[2]),
			jax.scipy.special.factorial(2+self.states-perception[6]-perception[5]-perception[4]-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-5)/jax.scipy.special.factorial(7-perception[6]-perception[5]-perception[4]-perception[3]-perception[2]),
			jax.scipy.special.factorial(1+self.states-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-6)/jax.scipy.special.factorial(7-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2]),
			jax.scipy.special.factorial(self.states-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-7)/jax.scipy.special.factorial(7-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2]),
			jax.scipy.special.factorial(-1+self.states-perception[9]-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-8)/jax.scipy.special.factorial(7-perception[9]-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2]),
			jax.scipy.special.factorial(-2+self.states-perception[10]-perception[9]-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-9)/jax.scipy.special.factorial(7-perception[10]-perception[9]-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2]),
			jax.scipy.special.factorial(-3+self.states-perception[11]-perception[10]-perception[9]-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-10)/jax.scipy.special.factorial(7-perception[11]-perception[10]-perception[9]-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2]),
			jax.scipy.special.factorial(-4+self.states-perception[12]-perception[11]-perception[10]-perception[9]-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2])/jax.scipy.special.factorial(self.states-11)/jax.scipy.special.factorial(7-perception[12]-perception[11]-perception[10]-perception[9]-perception[8]-perception[7]-perception[6]-perception[5]-perception[4]-perception[3]-perception[2]),
		]))
		position = jnp.zeros(self_state.shape).astype(jnp.int32)
		position = jnp.concatenate((jnp.array([jnp.indices(self_state.shape)[0]]), jnp.array([jnp.indices(self_state.shape)[1]])), axis=0)
		
		indices = (self_state * math.comb(7+self.states, 8)) + jnp.sum(indexcontributions[0:self.states-1], axis=0).astype(jnp.int32)
		state = self.srt[self.time[0], position[0], position[1], indices]
		self.time += jnp.array([1])
		return state.reshape(state.shape[0], -1, 1)

	@nnx.jit
	def update_srt(self, srt) -> None:
		"""Update the SRT of a SICA from an array input.

		Args:
			srt: An NDArray with 4 dimensions and last dimension of size 18.

		"""
		assert len(srt.shape) == 4, f"Expected 4 dimensions in SRT, received {len(srt.shape)}"
		assert (self.states * math.comb(7+self.states, 8) == srt.shape[3]), f"Expected {self.states * math.comb(7+self.states, 8)} indices in SRT ruleset, found {srt.shape[3]}"
		self.srt = srt

class NSICA(CA):
	"""Generalized (N-State) Spacetime-Inhomogeneous Cellular Automata."""

	def __init__(self, time, rngs: nnx.Rngs, *, states: int = 2, metrics_fn: Callable = metrics_fn):
		"""Initialize N-state SICA."""
		assert (states <= 12), f"Failed to allocate {states * math.comb(7+states, 8)} array slots for individual SRT ruleset allocation.\nEnsure that states <= 12; received value: {states}"
		assert (states >= 2), f"Number of SRT states must be at least 2 (received value: {states})"
		perceive = NSICAPerceive(rngs=rngs, states=states)
		update = NSICAUpdate(time=time, rngs=rngs, states=states)
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
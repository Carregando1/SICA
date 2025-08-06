"""Cellular Automata module for Time CAs."""

from collections.abc import Callable
from functools import partial

from flax import nnx
from jax import Array

from cax_sica.cax_new.core.perceive import Perceive
from cax_sica.cax_new.core.update import Update
from cax_sica.cax_new.types import Input, Metrics, Perception, State


def metrics_fn(next_state: State, state: State, perception: Perception, input: Input) -> Metrics:
	"""Metrics function returning the state.

	Args:
		next_state: Next state.
		state: Current state.
		perception: Perception.
		input: Input.

	Returns:
		A PyTree of metrics.

	"""
	return next_state


class CA(nnx.Module):
	"""Cellular Automata class."""

	def __init__(self, perceive: Perceive, update: Update, *, metrics_fn: Callable = metrics_fn):
		"""Initialize the CA.

		Args:
			perceive: Perception module.
			update: Update module.
			metrics_fn: Metrics function.

		"""
		self.perceive = perceive
		self.update = update
		self.metrics_fn = metrics_fn

	@nnx.jit
	def step(self, state: State, time: int = 0, input: Input | None = None) -> tuple[State, Metrics]:
		"""Perform a single step.

		Args:
			state: Current state.
			time: Current time. Time is always placed after state
			input: Optional input.

		Returns:
			Updated state.

		"""
		perception = self.perceive(state)
		next_state = self.update(state, time, perception, input)
		assert state.shape == next_state.shape, f"Shape of input state {state.shape} is inconsistent with shape of output state {next_state.shape}"
		time += 1
		return next_state, time, self.metrics_fn(next_state, state, perception, input)

	@partial(nnx.jit, static_argnames=("num_steps", "input_in_axis"))
	def __call__(
		self,
		state: State,
		time: int = 0,
		input: Input | None = None,
		*,
		num_steps: int = 1,
		input_in_axis: int | None = None,
	) -> tuple[State, Metrics]:
		"""Run the CA for multiple steps.

		Args:
			state: Initial state.
			time: Initial time, defaults to 0
			input: Optional input.
			num_steps: Number of steps to run.
			input_in_axis: Axis for input if provided for each step.

		Returns:
			Final state and all intermediate metrics.

		"""
		def step(carry: tuple[CA, State, int], input: Input | None) -> tuple[tuple[CA, State], State]:
			ca, state, time = carry
			state, time, metrics = ca.step(state, time, input)
			return (ca, state, time), metrics
		#Issue here
		(_, state, time), metrics = nnx.scan(
			step,
			in_axes=(nnx.Carry, input_in_axis),
			length=num_steps,
		)((self, state, time), input)

		return state, metrics

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB.

		Args:
			state: An array with two spatial/time dimensions.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		raise NotImplementedError

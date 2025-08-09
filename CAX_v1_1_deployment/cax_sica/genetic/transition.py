#Transition: Attempt 2 version.

#%pip install -U "jax[cuda12]"
#%pip install -U "cax"
import jax
import jax.numpy as jnp
from flax import nnx
from cax_sica.cax_new.systems.sica import sica


def transition(ic, srt, steps):
	"""
	transition.py: Takes in an IC and SRT and outputs the state after {steps} steps.
	ic: a 2D square array with only 0s and 1s.
	srt: a 4D array with dims (time, width, height, 18), the 18 being 2 cell states * 9 neighbor states.
	steps: The number of steps the SICA is run. Must be less than srt.shape[2].

	Attempt 2.
	"""

	seed = 0
	num_steps = steps
	time = 0
	rngs = nnx.Rngs(seed)

	ca = sica(rngs=rngs)
	state_init = jnp.zeros((ic.shape[0]+2, ic.shape[1]+2, 1)).at[1:-1, 1:-1].set(ic.reshape(ic.shape[0], -1, 1))
	srt = jnp.zeros((srt.shape[0], srt.shape[1]+2, srt.shape[2]+2, srt.shape[3])).at[:, 1:-1, 1:-1, :].set(srt)
	ca.update.update_srt(srt=srt)
	assert num_steps <= srt.shape[0] or not strict, f"Requested time {num_steps} is out of bounds, max time {srt.shape[0]}"
	state_final, states = ca(state=state_init, time=time, num_steps=num_steps)
	states = jnp.concatenate([(state_init[1:-1, 1:-1].astype(int)).reshape(1, ic.shape[0], -1), (states[:, 1:-1, 1:-1].astype(int)).reshape(states.shape[0], ic.shape[0], -1)])
	return states
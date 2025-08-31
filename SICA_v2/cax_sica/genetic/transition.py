#%pip install -U "jax[cuda12]"
#%pip install -U "cax"
import jax
import jax.numpy as jnp
from flax import nnx
from sica2s import sica

def transition(ic, srt, steps):
	"""
	transition.py: Takes in an IC and SRT and outputs the state after {steps} steps.
	ic: a 2D square array with only 0s and 1s.
	srt: a 4D array with dims (time, width, height, 18), the 18 being 2 cell states * 9 neighbor states.
	steps: The number of steps the SICA is run. Must be less than srt.shape[0].
	"""

	seed = 0
	num_steps = steps
	time = 0
	rngs = nnx.Rngs(seed)

	ca = sica(time=time, rngs=rngs)
	state_init = jnp.zeros((ic.shape[0]+2, ic.shape[1]+2, 1)).at[1:-1, 1:-1].set(ic.reshape(ic.shape[0], -1, 1))

	#In update 2.1 see if you can cut out the deadweight extra 2 in each dimension by changing the padding from CIRCULAR to VALID or SAME.
	srt = jnp.zeros((srt.shape[0], srt.shape[1]+2, srt.shape[2]+2, srt.shape[3])).at[:, 1:-1, 1:-1, :].set(srt)
	ca.update.update_srt(srt=srt)
	assert num_steps <= srt.shape[0], f"Requested time {num_steps} is out of bounds, max time {srt.shape[0]}"
	state_final, states = ca(state=state_init, num_steps=num_steps)
	states = jnp.concatenate([(state_init[1:-1, 1:-1].astype(int)).reshape(1, ic.shape[0], -1), (states[:, 1:-1, 1:-1].astype(int)).reshape(states.shape[0], ic.shape[0], -1)])
	return states
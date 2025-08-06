"""
transition.py:

perform transitions for states and the corresponding rules

SICA Decomp Version
"""
#%pip install -U "jax[cuda12]"
#%pip install -U "cax"


import numpy as np
from numpy import int32
from numpy._typing import NDArray
import jax
import jax.numpy as jnp
from flax import nnx
from cax.systems import *
from cax.systems.life import Life

"""Attempt 1 code."""

def bsfromarray(rules):
	#Takes in a JNPArray from the RuleSet class and converts it to BS notation
	converter = jnp.array([
		[8,0,0,0],
		[7,1,0,1],
		[6,2,0,2],
		[5,3,0,3],
		[4,4,0,4],
		[3,5,0,5],
		[2,6,0,6],
		[1,7,0,7],
		[0,8,0,8],
		[8,0,1,9],
		[7,1,1,10],
		[6,2,1,11],
		[5,3,1,12],
		[4,4,1,13],
		[3,5,1,14],
		[2,6,1,15],
		[1,7,1,16],
		[0,8,1,17],
	])
	strs = ["", ""]
	for i in range (rules.shape[0]):
		assert len(jnp.nonzero(jnp.sum(converter[0:, :-1] == rules[i][:-1], axis=1) == 3)[0]), "State ["+rules[i][0].__str__()+", "+rules[i][1].__str__()+", "+rules[i][2].__str__()+"] in RuleSet is invalid: ensure that neighbors add to 8 and state is either 0 or 1"
		num = converter[jnp.nonzero(jnp.sum(converter[0:, :-1] == rules[i][:-1], axis=1) == 3)[0][0]][3]
		if rules[i][3]: strs[int(jnp.floor(num/9))] += (num%9).__str__() 
	return "B"+strs[0]+"/S"+strs[1]

def transition(states, rules):
	seed = 0
	num_steps = 1
	key = jax.random.key(seed) #unnecessary?
	rngs = nnx.Rngs(seed)

	ca = Life(rngs=rngs)
	
	res = np.zeros((states.shape[0], states.shape[1]))
	icborder = jnp.zeros((states.shape[0]+2, states.shape[1]+2)).at[1:-1, 1:-1].set(states)
	#simulate every cell in ic, in a 5x5 to avoid errors, O(n^2)
	#Modify this portion so that all cells are run in parallel
	for i in range (1, states.shape[0] + 1):
		for j in range (1, states.shape[1] + 1):
			ca.update.update_birth_survival_from_string(bsfromarray(rules[i-1][j-1]))
			state_init = jnp.zeros((5, 5, 1)).at[1:-1, 1:-1, 0].set(icborder[i-1:i+2, j-1:j+2])
			state_final, stateslog = ca(state_init, num_steps=num_steps)
			res[i-1, j-1] = state_final.reshape(state_final.shape[1], -1)[2][2]
	return res.astype(int)
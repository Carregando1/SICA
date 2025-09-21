#Mutation: Degeneralized to strictly 2 state SICA.

from typing import List

import numpy as np
from numpy._typing import NDArray

from .state import CurrentState
import math

"""Mutation classes except for MutationSet no longer needed; replaced by list objects."""

class MutationSet:
    """
    A class storing a vector of ICMutation objects and SRTMutation objects, that is, a set of mutations being applied at once.
    """
    def __init__(self):
        self.ic_mutations = []
        self.srt_mutations = []

class Mutator:
    def mutate(self, state: CurrentState) -> tuple[CurrentState, MutationSet]:
        """
        This function should return a new mutation that we could try.
        """
        raise NotImplementedError

    def init_state(self) -> CurrentState:
        """
        This function should return some initial configuration that our genetic
        algorithm can work with.
        """
        raise NotImplementedError

"""
RulesetMutator class is removed in the generalization update. Use ArbitraryRulesetMutator instead.
"""

class ArbitraryRulesetMutator(Mutator):
    """
    Starts off with an initial condition consisting of some finite set of rules,
    and then arbitrarily mutates rules.
    """

    def __init__(
        self,
        grid_size: int = 32,
        states: int = 2,
        state_init_p: List[float] | None = None,
        mutate_p: float = 1 / (32**2),
        rule_mutate_p: float = 1 / 3,
        strict: bool = False,
    ):
        """
        Initializes the arbitrary ruleset mutator.

        :param List[] rules: the set of rules to apply to the grid.

        :param int grid_size: the size of the grid that our mutator will work on.

        :param List[float] state_init_p: the probability that we should choose a particular state when initializing (it MUST have the same length as the number of states in all the elements).

        :param float mutate_p: the probability that we mutate any cell, which include the cells representing the initial configuration space and the cells for each ruleset.

        :param float rule_mutate_p: the probability that we mutate a given sub-rule, given that we have selected the rule for modification.

        """
        # initialize grid size
        self.grid_size = grid_size

        # initialize states
        self.states = states

        # initialize strict mode: strict = SRT mutations must correspond to the locations of the IC mutations
        self.strict = strict

        assert (
            state_init_p is None or len(state_init_p) == states
        ), "The length of the state probs array does not equal the number of valid states!"
        self.state_init_p = state_init_p

        self.mutate_p = mutate_p

        self.rule_mutate_p = rule_mutate_p

    def init_state(self) -> CurrentState:
        # initialize the rules based on the rule indices above.
        rules = np.zeros((self.grid_size - 1, self.grid_size, self.grid_size, self.states * math.comb(7+self.states, 8)))
        initial = np.random.choice(
            range(self.states),
            size=(self.grid_size, self.grid_size),
            p=self.state_init_p,
        )
        return CurrentState(rules=rules, initial=initial)



    def mutate(self, state: CurrentState) -> tuple[CurrentState, MutationSet]:
        """
        Mutates an existing state to a new state stochastically.
        """
        new_state = CurrentState(
            initial=state.initial.copy(),
            rules=state.rules.copy(),
        )

        # this logic is shared with RulesetMutator
        mutations = MutationSet()
        
        """Use of ICMutation and SRTMutation objects are now obsolete to remove all for loops."""

        #Note that this program is different from the original genetic algorithm in that the ruleset mutations need not be in the same place as the initial mutations!

        #Mutate initials.

        new_state.initial = state.initial + ((np.random.rand(state.initial.shape[0],state.initial.shape[1]) < self.mutate_p)*np.random.randint(1, self.states))
        new_state.initial = new_state.initial%self.states
        a = np.nonzero(new_state.initial != state.initial)
        mutations.ic_mutations = np.concatenate((np.transpose(a), state.initial[a].reshape(-1,1), new_state.initial[a].reshape(-1,1)), axis=1).tolist()

        #Mutate SRT.
        if (self.strict): 
            #Chooses SRT cells to mutate with probability mutate_p,
            #then chooses individual rules within the chosen SRT cells to mutate with probability rule_mutate_p
            new_state.rules = state.rules + (np.random.rand(state.rules.shape[0], state.rules.shape[1], state.rules.shape[2], 1) < self.mutate_p) * (np.random.rand(state.rules.shape[0], state.rules.shape[1], state.rules.shape[2], state.rules.shape[3]) < self.rule_mutate_p) * np.random.randint(1, self.states, size=state.rules.shape)
        else:
            #Selects arbitrary SRT cells with probability rule_mutate_p
            new_state.rules = state.rules + (np.random.rand(state.rules.shape[0],state.rules.shape[1],state.rules.shape[2],state.rules.shape[3]) < self.rule_mutate_p)*np.random.randint(1, self.states, size=state.rules.shape)
        new_state.rules = new_state.rules%self.states
        a = np.nonzero(new_state.rules != state.rules)
        mutations.srt_mutations = np.concatenate((np.transpose(a), state.rules[a].reshape(-1,1), new_state.rules[a].reshape(-1,1)), axis=1).tolist()

        return (new_state, mutations)

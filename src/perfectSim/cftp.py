'''
CFTP algorithm.
'''

import numpy as np
from src.models.baseModel import MonotoneModel

def monotone_cftp(model: MonotoneModel, B: int = 1, k: int = 32, D_init: int = 1):
    '''
    Perform CFTP for given model to produce B samples
    samples from the stationary dist
    '''
    base_sequence = np.random.SeedSequence()
    keys = np.array(base_sequence.spawn(B))

    active = np.arange(B)
    samples = np.zeros((B, model.n), dtype=np.int8)

    D = D_init
    safe_counter = 1

    while active.size > 0:
        A = active.size

        states = np.empty((2*A, model.n), dtype=np.int8)

        states[:A] = model.top
        states[A:] = model.bottom

        for d in range(D, 0, -1):
            safe_counter = d * k * 1000 # RNG must be sufficiently spaced to ensure independence

            active_keys = keys[active]
            sites, unifs = model.randomness(safe_counter, active_keys, k)

            stacked_sites = np.vstack((sites, sites))
            stacked_unifs = np.vstack((unifs, unifs))

            r = (stacked_sites, stacked_unifs)

            states = model.apply(states, r)

        coalesced_mask = (states[:A] == states[A:]).all(axis=1)

        if coalesced_mask.any():
            finished_indices = active[coalesced_mask]
            samples[finished_indices] = states[:A][coalesced_mask]
            active = active[~coalesced_mask]

        D *= 2

    return samples

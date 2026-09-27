'''
Bounding chain CFTP for non-monotone chains.
'''

import numpy as np
from src.models.baseModel import BoundingModel

def bounding_cftp(model: BoundingModel, B: int = 1, seed: int = None, k: int = 32, D_init: int = 1):
    '''
    Perform CFTP using bounding chain to produce B samples.
    '''
    base_sequence = np.random.SeedSequence(seed)
    keys = np.array(base_sequence.spawn(B))

    active = np.arange(B)
    samples = np.zeros((B, model.n), dtype=np.int8)

    D = D_init
    safe_counter = 1

    while active.size > 0:
        A = active.size

        states = np.empty((A, model.n), dtype=np.int8)

        states[:A] = model.bounding_initial

        for d in range(D, 0, -1):
            safe_counter = d * k * 1000

            active_keys = keys[active]
            r = model.randomness(safe_counter, active_keys, k)

            states = model.apply(states, r)

        coalesced_mask = (states != model.unknown).all(axis=1)

        if coalesced_mask.any():
            finished_indices = active[coalesced_mask]
            samples[finished_indices] = states[coalesced_mask]
            active = active[~coalesced_mask]

        D *= 2

    return samples
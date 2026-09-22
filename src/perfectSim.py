'''
CFTP algorithm.
'''

import numpy as np
from src.baseModel import BaseModel

def cftp(model: BaseModel, B: int = 1, k: int = 32, D_init: int = 1):
    '''
    Perform CFTP for given model to produce B samples
    samples from the stationary dist
    '''
    keys = np.random.randint(0, 2**31, size=B, dtype=np.int32)

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

            for i in range(A): # In future need to change to apply_batch that acts on the whole (2B,n) array at once
                key = keys[active[i]]

                r = model.randomness(safe_counter, key, k)

                states[i] = model.apply(states[i], r)
                states[i + A] = model.apply(states[i + A], r)

        coalesced_mask = (states[:A] == states[A:]).all(axis=1)

        if coalesced_mask.any():
            finished_indices = active[coalesced_mask]
            samples[finished_indices] = states[:A][coalesced_mask]
            active = active[~coalesced_mask]

        D *= 2
    return samples
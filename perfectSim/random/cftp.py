'''
CFTP algorithm.
'''

import numpy as np
from perfectSim.models.baseModel import MonotoneModel

def monotone_cftp(model: MonotoneModel, B: int = 1, trace: bool = False,  seed: int = None, k: int = 32, D_init: int = 1):
    '''
    Perform CFTP for given model to produce B samples
    samples from the stationary dist.
    '''
    base_sequence = np.random.SeedSequence(seed)
    keys = np.array(base_sequence.spawn(B))

    active = np.arange(B)
    samples = np.zeros((B, model.n), dtype=np.int8)

    D = D_init
    safe_counter = 1

    trace_dict = {b: None for b in range(B)} if trace else None

    while active.size > 0:
        A = active.size

        states = np.empty((2*A, model.n), dtype=np.int8)

        states[:A] = model.top
        states[A:] = model.bottom

        if trace:
            attempt_trace_top = [states[:A].copy()]
            attempt_trace_bottom = [states[A:].copy()]

        for d in range(D, 0, -1):
            safe_counter = d * k * 1000 # RNG must be sufficiently spaced to ensure independence

            active_keys = keys[active]
            sites, unifs = model.randomness(safe_counter, active_keys, k)

            stacked_sites = np.vstack((sites, sites))
            stacked_unifs = np.vstack((unifs, unifs))

            r = (stacked_sites, stacked_unifs)

            states = model.apply(states, r)

            if trace:
                attempt_trace_top.append(states[:A].copy())
                attempt_trace_bottom.append(states[A:].copy())

        coalesced_mask = (states[:A] == states[A:]).all(axis=1)

        if coalesced_mask.any():
            finished_indices = active[coalesced_mask]
            samples[finished_indices] = states[:A][coalesced_mask]

            if trace:
                arr_top = np.array(attempt_trace_top)
                arr_bottom = np.array(attempt_trace_bottom)

                local_indices = np.where(coalesced_mask)[0]

                for i in local_indices:
                    orig_idx = active[i]
                    trace_dict[orig_idx] = {
        'top': arr_top[:, i, :],
        'bottom': arr_bottom[:, i, :]
    }

            active = active[~coalesced_mask]

        D *= 2

    if trace:
        return samples, trace_dict
    
    return samples

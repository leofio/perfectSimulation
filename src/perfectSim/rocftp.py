'''
Ro-CFTP algorithm.
'''

import numpy as np
from typing import Optional, Tuple
from src.baseModel import BaseModel

def _simulate_block(
    model: BaseModel, 
    stream_counter: int, 
    keys: np.ndarray, 
    k: int, 
    X: Optional[np.ndarray] = None
) -> Tuple[bool, np.ndarray, int]:
    '''
    Helper to pull randomness and apply a block of size k.
    If X is provided it is stepped with the rest of the block.
    '''
    sites, unifs = model.randomness(stream_counter, keys, k)

    if X is None:
        states = np.vstack((model.top, model.bottom))
        r = (np.vstack((sites, sites))), (np.vstack((unifs, unifs)))
    else:
        states = np.vstack((model.top, model.bottom, X))
        r = (np.vstack((sites, sites, sites))), (np.vstack((unifs, unifs, unifs)))

    states = model.apply(states, r)
    is_coalesced = model.equal(states[0], states[1])

    return is_coalesced, states, stream_counter + k

    

def rocftp_fixed(model: BaseModel, k: int, B: int =1, key: int = 1) -> np.ndarray:
    '''
    Read-only CFTP algorithm with fixed block sizes.
    Produces B exact samples using stricly forward stream of randomness.

    k: Block size. Recommended large enough that P(coalescence in K steps) > 0.
    '''
    samples = np.zeros((B, model.n), dtype=np.int8)
    stream_counter = 1

    keys = np.array([key])

    S = None
    while True:
        is_coalesced, states, stream_counter = _simulate_block(model, stream_counter, keys, k)
        if is_coalesced:
            S = states[0]
            break

    for b in range(B):
        X = S.copy()

        while True:
            is_coalesced, states, stream_counter = _simulate_block(model, stream_counter, keys, k, X)

            if is_coalesced:
                samples[b] = X
                S = states[0]
                break
            else:
                X = states[2]

    return samples
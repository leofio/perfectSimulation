'''
Throwaway trajectory generator for building/testing the visualiser.

This runs the model's own chain forward with k=1 (one site update per call).
It is NOT a perfect sample: the states are only stationary in the long run.
It exists so the renderer has realistic (T, n) data before the samplers can log.
'''

from typing import Optional, Tuple
import numpy as np


def forward_trajectory(model, x0: Optional[np.ndarray] = None, steps: int = 200,
                       stride: int = 1, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Returns (states, updated):
        states:  (steps // stride + 1, model.n) int8, states[0] = x0
        updated: same length; updated[t] = index updated at the last step before
                 frame t (-1 for frame 0). Feed to animate(highlight=...) for vertex models.

    x0 defaults to model.bounding_initial (all-unknown) for bounding models,
    else model.bottom. Starting a bounding model from all-unknown and animating
    is a nice way to watch the unknown sites resolve.
    '''
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = model.bounding_initial if hasattr(model, 'bounding_initial') else model.bottom
    x = np.asarray(x0, dtype=np.int8).reshape(1, -1).copy()
    if x.shape[1] != model.n:
        raise ValueError(f'x0 has {x.shape[1]} entries, model has {model.n} variables')

    states, updated = [x[0].copy()], [-1]
    last = -1
    for step in range(1, steps + 1):
        last = int(rng.integers(0, model.n))
        r = (np.array([[last]], dtype=np.int32), np.array([[rng.random()]]))
        x = np.array(model.apply(x, r), dtype=np.int8)
        if step % stride == 0:
            states.append(x[0].copy())
            updated.append(last)
    return np.array(states), np.array(updated)

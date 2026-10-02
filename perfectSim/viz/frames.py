'''
Frames: everything the renderer needs to draw one picture, as plain arrays.

    values        (n_sites,)    per-vertex state, or None (draw neutral vertices)
    ghost_values  (n_boundary,) per-boundary-vertex value (FREE = unconstrained)
    edge_on       (n_edges,)    in [0, 1]; 1 = solid, 0 = faint

Edge strength comes either straight from an edge-state array (random-cluster
states: open/closed) or from an 'edge rule' derived from vertex values.
'''

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from perfectSim.lattice.lattice import FREE, Lattice


def ghost_values_from_boundary(lat: Lattice, boundary: Optional[dict]) -> np.ndarray:
    '''
    Convert a model-style boundary dict {global ghost id: value} into an array of
    length n_boundary. Unspecified ghosts get FREE.
    '''
    gv = np.full(lat.n_boundary, FREE, dtype=np.int32)
    for g, s in (boundary or {}).items():
        if not (lat.n_sites <= g < lat.null):
            raise ValueError(f'boundary key {g} is not a ghost id (valid: {lat.n_sites}..{lat.null - 1})')
        gv[g - lat.n_sites] = s
    return gv


def _full(lat, values, ghost_values):
    if not lat.n_boundary:
        return values
    gv = ghost_values if ghost_values is not None else np.full(lat.n_boundary, FREE)
    return np.concatenate([values, gv])


# ---- edge rules: (lattice, values, ghost_values) -> (n_edges,) floats in [0, 1] ----------

def rule_solid(lat, values, ghost_values):
    '''Every edge solid.'''
    return np.ones(lat.n_edges)


def rule_aligned(lat, values, ghost_values):
    '''Edge solid iff both endpoints hold the same value (shows domains / Potts clusters).'''
    if values is None:
        return np.ones(lat.n_edges)
    full = _full(lat, values, ghost_values)
    return (full[lat.edges[:, 0]] == full[lat.edges[:, 1]]).astype(float)


EDGE_RULES = {'solid': rule_solid, 'aligned': rule_aligned}


def resolve_rule(rule) -> Callable:
    if callable(rule):
        return rule
    if rule in EDGE_RULES:
        return EDGE_RULES[rule]
    raise ValueError(f'edge_rule must be a callable or one of {sorted(EDGE_RULES)}, got {rule!r}')


@dataclass(frozen=True)
class Frame:
    values: Optional[np.ndarray]
    ghost_values: Optional[np.ndarray]
    edge_on: np.ndarray


def build_frame(lat, values, ghost_values, edge_state, rule) -> Frame:
    if values is not None and values.shape != (lat.n_sites,):
        raise ValueError(f'values must have shape ({lat.n_sites},), got {values.shape}')

    if edge_state is not None:
        if edge_state.shape != (lat.n_edges,):
            raise ValueError(f'edge_states must have shape ({lat.n_edges},), got {edge_state.shape}')
        edge_on = np.clip(edge_state.astype(float), 0.0, 1.0)
    else:
        edge_on = np.asarray(rule(lat, values, ghost_values), dtype=float)
        if edge_on.shape != (lat.n_edges,):
            raise ValueError(f'edge rule returned shape {edge_on.shape}, expected ({lat.n_edges},)')

    return Frame(values, ghost_values, edge_on)

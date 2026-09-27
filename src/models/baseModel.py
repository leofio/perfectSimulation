'''
Abstract base class for other model to build on.
'''

from abc import ABC, abstractmethod
import numpy as np
from src.lattice import Lattice

class BaseModel(ABC):
    def __init__(self, lattice: Lattice):
        self.lattice = lattice

    @property
    @abstractmethod
    def n(self) -> int:
        '''Number of variables in the system.'''
        pass

    @abstractmethod
    def apply(self, states: np.ndarray, r) -> np.ndarray:
        '''Apply transitions to the state with randomness r'''
        pass

    def equal(self, x: np.ndarray, y: np.ndarray) -> bool:
        '''Check if two states are equal'''
        return np.array_equal(x, y)

    def randomness(self, depth, keys, k):
                '''
                Get the randomness for k steps over a batch of keys.
                keys: (batch_size,) array of seed keys
                Returns sites and unifs of shape (batch_size, k)
                '''
                batch_size = len(keys)
                sites = np.empty((batch_size, k), dtype=np.int32)
                unifs = np.empty((batch_size, k), dtype=np.float64)
        
                for i, key in enumerate(keys):
                    rng = np.random.Generator(np.random.Philox(seed=key, counter=depth))
                    sites[i] = rng.integers(0, self.n, k)
                    unifs[i] = rng.random(k)
        
                return sites, unifs

class MonotoneModel(BaseModel):
    def __init__(self, lattice):
        super().__init__(lattice)

    @property
    @abstractmethod
    def bottom(self) -> np.ndarray:
        '''Lowest state in partial order.'''
        pass

    @property
    @abstractmethod
    def top(self) -> np.ndarray:
        '''Highest state in partial order.'''
        pass

    @abstractmethod
    def leq(self, x: np.ndarray, y: np.ndarray) -> bool:
        '''Check the partial order, x<=y'''
        pass

class BoundingModel(BaseModel):
    def __init__(self, lattice):
         super().__init__(lattice)

    @property
    def unknown(self):
        pass

    @property
    def bounding_initial(self): return np.full(self.n, self.unknown, dtype=np.int8)
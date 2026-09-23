'''
Abstract base class for other model to build on.
'''

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @property
    @abstractmethod
    def n(self) -> int:
        '''Number of sites in the system.'''
        pass

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
    def randomness(self, depth: int, key: int, k: int):
        '''Shared randomness for k steps'''
        pass

    @abstractmethod
    def apply(self, state: np.ndarray, r) -> np.ndarray:
        '''Apply transitions to the state with randomness r'''
        pass

    @abstractmethod
    def leq(self, x: np.ndarray, y: np.ndarray) -> bool:
        '''Check the partial order, x<=y'''
        pass

    def equal(self, x: np.ndarray, y: np.ndarray) -> bool:
        '''Check if two states are equal'''
        return np.array_equal(x, y)

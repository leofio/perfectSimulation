'''
Random Cluster model.
'''

import numpy as np
from src.lattice import Lattice
from baseModel import MonotoneModel
from collections import defaultdict

class MonotoneRandomCluster(MonotoneModel):
    def __init__(self, lattice: Lattice, p: float, q: float, boundary_partitions=None):
        super().__init__(lattice)
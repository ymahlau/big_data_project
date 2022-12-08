import math
import random
from typing import Tuple, List

# Select rows as 90%-train, 10%-test split
def generate_indices(n: int) -> Tuple[List[int], List[int]]:
    n_train = math.floor(0.9 * n)
    row_idx = list(range(n))
    row_idx_train = random.sample(row_idx, k=n_train)
    row_idx_test = list(set(row_idx) - set(row_idx_train))
    return row_idx_train, row_idx_test

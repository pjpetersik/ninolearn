import numpy as np

def print_header(string):
    print()
    print("#######################################################################")
    print(string)
    print("#######################################################################")
    print()

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)
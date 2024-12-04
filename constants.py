import numpy as np

def pg_coeffs():
    """Zero indexed by PG case."""

    a = np.array([170.0, 98.0, 61.0, 32.0, 21.0, 14.0])
    b = np.array([1.09, 0.98, 0.91, 0.81, 0.75, 0.68])
    c = np.array([24.0, 18.0, 12.0, 8.0, 6.0, 4.0])
    d = np.array([2.5, 1.8, 1.1, 0.72, 0.54, 0.36])
    return a, b, c, d
:q

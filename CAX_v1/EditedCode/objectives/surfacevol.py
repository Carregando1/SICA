from numpy import int32
from numpy._typing import NDArray
import numpy as np

DX = [
    (+1, 0, 0),
    (-1, 0, 0),
    (0, +1, 0),
    (0, -1, 0),
    (0, 0, +1),
    (0, 0, -1),
]

def surface_to_vol(grid: NDArray[int32]):
    """
    calculate the surface area to volume ratio for a 3x3 lattice. Optimized using numpy.
    """
    vol = np.count_nonzero(grid)
    surface = (np.count_nonzero(grid[0:-1,:,:] * (grid[1:,:,:] == 0)) + np.count_nonzero(grid[-1,:,:]) +
                np.count_nonzero(grid[1:,:,:] * (grid[0:-1:,:,:] == 0)) + np.count_nonzero(grid[0,:,:]) +
                np.count_nonzero(grid[:,0:-1,:] * (grid[:,1:,:] == 0)) + np.count_nonzero(grid[:,-1,:]) +
                np.count_nonzero(grid[:,1:,:] * (grid[:,0:-1,:] == 0)) + np.count_nonzero(grid[:,0,:]) +
                np.count_nonzero(grid[:,:,0:-1] * (grid[:,:,1:] == 0)) + np.count_nonzero(grid[:,:,-1]) +
                np.count_nonzero(grid[:,:,1:] * (grid[:,:,0:-1] == 0)) + np.count_nonzero(grid[:,:,0]))
    return - surface / vol

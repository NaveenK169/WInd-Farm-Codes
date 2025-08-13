import numpy as np

def meshing(lx, ly, lz, nx, ny, nz):
    # Calculate the spacing between each grid point along each axis
    dx = lx / nx  # Spacing in the x-direction
    dy = ly / ny  # Spacing in the y-direction
    dz = lz / nz  # Spacing in the z-direction

    # Generate arrays of grid points along each axis using np.arange
    # A small amount (0.0001) is added to the stop value to ensure the last point is included in the array
    x = np.arange(0, lx - dx + 0.0001, dx)  # Create x grid points from 0 to lx with spacing dx
    y = np.arange(0, ly - dy + 0.0001, dy)  # Create y grid points from 0 to ly with spacing dy
    z = np.arange(dz / 2, lz - dz / 2 + 0.0001, dz)  # Create z grid points from dz/2 to lz - dz/2 with spacing dz

    # Return the grid spacing and the generated grid points for each axis
    return dx, dy, dz, x, y, z

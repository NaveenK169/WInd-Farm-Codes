import numpy as np

def get_k3D(nx, ny, nz, dx, dy, dz):
    """
    This function calculates the 3D wave numbers (spatial frequencies) for a given 3D grid in 
    Fourier space and computes some scaling factors (deal values) based on the grid size and 
    domain lengths.
    
    Arguments:
    nx, ny, nz -- The number of grid points in the x, y, and z directions respectively.
    dx, dy, dz -- The grid spacing in the x, y, and z directions respectively.
    
    Returns:
    k1, k2, k3 -- 3D arrays of wave numbers for each direction (x, y, z).
    kxdeal, kydeal, kzdeal -- Scaling factors for each direction.
    """
    
    # Step 1: Calculate the total domain lengths in each direction (x, y, z)
    Lx = nx * dx  # Total length in the x-direction
    Ly = ny * dy  # Total length in the y-direction
    Lz = nz * dz  # Total length in the z-direction
    
    # Step 2: Generate wave numbers (spatial frequencies) in each direction (kx, ky, kz)
    # These are calculated based on the number of grid points and the domain lengths.
    # We use np.fft.fftfreq to generate frequency bins, which are scaled by 2*pi/L to get wave numbers.
    # We use np.fft.fftshift to makesure zero at the center.
    
    # Generate kx, ky, kz wave numbers corresponding to each spatial dimension
    kx = np.fft.fftshift(np.arange(-nx//2, nx//2)) * (2 * np.pi / Lx)
    ky = np.fft.fftshift(np.arange(-ny//2, ny//2)) * (2 * np.pi / Ly)
    kz = np.fft.fftshift(np.arange(-nz//2, nz//2)) * (2 * np.pi / Lz)
    
    # Step 3: Create 3D grid of wave numbers (k1, k2, k3) using np.meshgrid
    # This constructs the full 3D Fourier space grid of wave numbers from kx, ky, kz.
    k1, k2, k3 = np.meshgrid(kx, ky, kz, indexing='ij')
    # 'indexing="ij"' ensures the grid is indexed like in a typical 3D Cartesian grid, where:
    #   k1 corresponds to kx (x direction),
    #   k2 corresponds to ky (y direction),
    #   k3 corresponds to kz (z direction).
    
    # Step 4: Compute "deal" values based on the grid sizes and domain lengths
    # These scaling factors (kxdeal, kydeal, kzdeal) are fractions of the respective wave numbers.
    kxdeal = (1 / 3) * nx * (2 * np.pi / Lx)  # Deal value for x direction
    kydeal = (1 / 3) * ny * (2 * np.pi / Ly)  # Deal value for y direction
    kzdeal = (1 / 3) * nz * (2 * np.pi / Lz)  # Deal value for z direction
    
    return k1, k2, k3, kxdeal, kydeal, kzdeal



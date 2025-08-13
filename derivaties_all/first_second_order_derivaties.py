import numpy as np

def ddx(f, kx):
    """
    This function calculates the derivative of the input function 'f' with respect to x using
    Fourier transforms. It uses the wave numbers 'kx' and the Fast Fourier Transform (FFT)
    to compute the derivative in Fourier space, then transforms the result back to real space.
    
    Arguments:
    f -- Input function (numpy array), shape (nx, ny, nz).
    kx -- Wave numbers in the x-direction (numpy array), shape (nx,).
    
    Returns:
    dfdx -- Derivative of f with respect to x (numpy array), same shape as f.
    """
    
    # Step 1: Get the number of grid points in the x-direction (nx)
    nx = f.shape[0]
    
    # Step 2: Compute the Fourier transform of the input function 'f' along the x-axis
    fhat = np.fft.fft(f, axis=0)
    
    # Step 3: Multiply the Fourier transform by 1j * kx to compute the derivative in Fourier space
    ghat = 1j * kx * fhat  # Broadcasting kx across the other dimensions
    ## in pyhton imaginary unit is denoted by j
    
    # Step 4: Set the Nyquist component (halfway point in the array) to 0
    ghat[nx // 2] = 0
    
    # Step 5: Compute the inverse Fourier transform to return to real space
    dfdx = np.real(np.fft.ifft(ghat, axis=0))
    
    return dfdx



def ddy(f, ky):
    """
    This function calculates the derivative of the input function 'f' with respect to y using
    Fourier transforms. It uses the wave numbers 'ky' and the Fast Fourier Transform (FFT)
    to compute the derivative in Fourier space, then transforms the result back to real space.
    
    Arguments:
    f -- Input function (numpy array), shape (nx, ny, nz).
    ky -- Wave numbers in the y-direction (numpy array), shape (ny,).
    
    Returns:
    dfdy -- Derivative of f with respect to y (numpy array), same shape as f.
    """
    
    # Step 1: Get the number of grid points in the y-direction (ny)
    ny = f.shape[1]
    
    # Step 2: Compute the Fourier transform of the input function 'f' along the y-axis
    fhat = np.fft.fft(f, axis=1)
    
    # Step 3: Multiply the Fourier transform by 1i * ky to compute the derivative in Fourier space
    ghat = 1j * ky[None, :, None] * fhat  # Broadcasting ky across the other dimensions
    
    # Step 4: Set the Nyquist component (halfway point in the array) to 0
    ghat[:, ny // 2, :] = 0
    
    # Step 5: Compute the inverse Fourier transform to return to real space
    dfdy = np.real(np.fft.ifft(ghat, axis=1))
    
    return dfdy



def ddz(f, kz):
    """
    This function calculates the derivative of the input function 'f' with respect to z using
    Fourier transforms. It uses the wave numbers 'kz' and the Fast Fourier Transform (FFT)
    to compute the derivative in Fourier space, then transforms the result back to real space.
    
    Arguments:
    f -- Input function (numpy array), shape (nx, ny, nz).
    kz -- Wave numbers in the z-direction (numpy array), shape (nz,).
    
    Returns:
    dfdz -- Derivative of f with respect to z (numpy array), same shape as f.
    """
    
    # Step 1: Get the number of grid points in the z-direction (nz)
    nz = f.shape[2]
    
    # Step 2: Compute the Fourier transform of the input function 'f' along the z-axis
    fhat = np.fft.fft(f, axis=2)
    
    # Step 3: Multiply the Fourier transform by 1i * kz to compute the derivative in Fourier space
    ghat = 1j * kz[None, None, :] * fhat  # Broadcasting kz across the other dimensions
    
    # Step 4: Set the Nyquist component (halfway point in the array) to 0
    ghat[:, :, nz // 2] = 0
    
    # Step 5: Compute the inverse Fourier transform to return to real space
    dfdz = np.real(np.fft.ifft(ghat, axis=2))
    
    return dfdz



def ddz_cd2(f, z):
    """
    This function calculates the derivative of the input function 'f' with respect to z using
    a second-order central difference scheme for the z-direction. The function 'f' is assumed
    to be a 3D numpy array. The grid spacing 'z' is assumed to be a 1D array of size (nz,).
    
    Arguments:
    f -- Input function (numpy array), shape (nx, ny, nz).
    z -- Grid points in the z-direction (numpy array), shape (nz,).
    
    Returns:
    df -- Derivative of f with respect to z (numpy array), same shape as f.
    """
    
    # Get the shape of the input array
    nx, ny, nz = f.shape
    
    # Initialize the derivative array
    df = np.zeros_like(f)
    
    # Compute the grid spacing dz
    dz = np.zeros((1, 1, nz))
    dz[0, 0, 0] = z[1] - z[0]  # First grid spacing
    for k in range(1, nz - 1):
        dz[0, 0, k] = z[k + 1] - z[k - 1]  # Central difference for interior points
    dz[0, 0, nz - 1] = z[nz - 1] - z[nz - 2]  # Last grid spacing
    
    # Compute the central difference in the z-direction
    for i in range(nx):
        for j in range(ny):
            df[i, j, 0] = (f[i, j, 1] - f[i, j, 0]) / dz[0, 0, 0]  # Forward difference for the first point
            df[i, j, 1:nz - 1] = (f[i, j, 2:nz] - f[i, j, 0:nz - 2]) / dz[0, 0, 1:nz - 1]  # Central difference for interior points
            df[i, j, nz - 1] = (f[i, j, nz - 1] - f[i, j, nz - 2]) / dz[0, 0, nz - 1]  # Backward difference for the last point
    
    return df



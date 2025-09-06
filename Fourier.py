import numpy as np

def laplacian_fft(Z, dx, dy):
    """
    Pseudo-spectral Laplacian via FFT (periodic BC).
    """
    nx, ny = Z.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Z_hat = np.fft.fftn(Z)
    lap_hat = -(KX**2 + KY**2) * Z_hat
    return np.fft.ifftn(lap_hat).real

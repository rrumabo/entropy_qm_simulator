import numpy as np
from numpy.fft import fft, ifft, fftfreq

def solve_entropy_schrodinger(psi0, L, ħ, m, γ, dt, t_max, eps=1e-12):
    """
    Simulate entropy-regularized Schrödinger evolution of a quantum wavefunction.
    
    Parameters:
    - psi0: np.ndarray, initial wavefunction (complex)
    - L: float, system length
    - ħ: float, reduced Planck constant
    - m: float, particle mass
    - γ: float, entropy coupling strength
    - dt: float, time step
    - t_max: float, total simulation time
    - eps: float, entropy regularization cutoff

    Returns:
    - ψ_t: np.ndarray, evolved wavefunction over time (shape: [num_steps, N])
    - x: np.ndarray, position grid
    - times: np.ndarray, time points
    """
    N = len(psi0)
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    k = 2 * np.pi * fftfreq(N, d=dx)
    T_k = (ħ**2 / (2 * m)) * k**2

    ψ = psi0 / np.linalg.norm(psi0)
    num_steps = int(t_max / dt)
    ψ_t = np.zeros((num_steps, N), dtype=complex)
    times = np.linspace(0, t_max, num_steps)

    for n in range(num_steps):
        ψ_k = fft(ψ)
        H0_ψ = np.real(ifft(T_k * ψ_k))

        ρ = np.abs(ψ)**2
        entropy_term = γ * (np.log(np.maximum(ρ, eps)) + 1) * ψ

        ψ = ψ - (1j * dt / ħ) * H0_ψ + dt * entropy_term
        ψ /= np.linalg.norm(ψ)
        ψ_t[n] = ψ

    return ψ_t, x, times

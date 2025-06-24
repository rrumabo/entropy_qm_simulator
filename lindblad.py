import numpy as np

def evolve_lindblad_density_matrix(rho0, H, x_op, alpha, dt, t_max, ħ=1.0):

    N = rho0.shape[0]
    num_steps = int(t_max / dt)
    times = np.linspace(0, t_max, num_steps)
    rho_t = np.zeros((num_steps, N, N), dtype=complex)
    rho = rho0.copy()

    for i in range(num_steps):
        commutator = -1j / ħ * (H @ rho - rho @ H)
        x_rho_x = x_op @ rho @ x_op
        x2_rho = x_op @ x_op @ rho
        rho_x2 = rho @ x_op @ x_op
        lindblad_term = alpha * (x_rho_x - 0.5 * (x2_rho + rho_x2))

        rho += dt * (commutator + lindblad_term)
        rho = (rho + rho.conj().T) / 2  # Keep Hermitian
        rho_t[i] = rho

    return rho_t, times

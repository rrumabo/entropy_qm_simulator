import numpy as np
import matplotlib.pyplot as plt
from entropy_qm_simulator.simulator import solve_entropy_schrodinger
from entropy_qm_simulator.lindblad import evolve_lindblad_density_matrix
from benchmarks.utils import compute_entropy, compute_purity, compute_localization_width

# Set up grid and initial state
ħ, m, L, N = 1.0, 1.0, 10.0, 256
x = np.linspace(0, L, N, endpoint=False)
dx = L / N
psi0 = np.exp(-(x - L/2)**2)
psi0 /= np.linalg.norm(psi0)
rho0 = np.outer(psi0, psi0.conj())

# Hamiltonian: finite-diff kinetic operator
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
T_k = (ħ**2 / (2 * m)) * k**2
H = np.diag(np.real(np.fft.ifft(np.fft.fft(np.eye(N)) * T_k[:, None], axis=0)))

# Position operator
x_op = np.diag(x)

# Parameters
gamma = 0.1
dt = 0.001
t_max = 1.0

# Run both simulations
ψ_t, _, times = solve_entropy_schrodinger(psi0, L, ħ, m, gamma, dt, t_max)
rho_t, _ = evolve_lindblad_density_matrix(rho0, H, x_op, gamma, dt, t_max)

# Metrics
entropy_entropy = [compute_entropy(np.outer(ψ, ψ.conj())) for ψ in ψ_t]
entropy_lindblad = [compute_entropy(rho) for rho in rho_t]
purity_lindblad = [compute_purity(rho) for rho in rho_t]

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(times, entropy_entropy, label='Entropy Schrödinger')
plt.plot(times, entropy_lindblad, label='Lindblad')
plt.xlabel('Time')
plt.ylabel('von Neumann Entropy')
plt.legend()
plt.title('Entropy Evolution')
plt.tight_layout()
plt.show()

import argparse
import numpy as np
import matplotlib.pyplot as plt
from entropy_qm_simulator.simulator import solve_entropy_schrodinger

def main():
    parser = argparse.ArgumentParser(description="Entropy Schrödinger simulator")
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--tmax', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--L', type=float, default=10.0)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    ħ = 1.0
    m = 1.0
    x = np.linspace(0, args.L, args.N, endpoint=False)
    psi0 = np.exp(-(x - args.L/2)**2)
    psi0 /= np.linalg.norm(psi0)

    ψ_t, x, times = solve_entropy_schrodinger(psi0, args.L, ħ, m, args.gamma, args.dt, args.tmax)

    if args.plot:
        import seaborn as sns
        sns.set()
        from benchmarks.utils import compute_entropy
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(np.abs(ψ_t)**2, extent=[x[0], x[-1], times[-1], times[0]], aspect='auto', cmap='magma')
        ax[0].set_title("Probability density over time")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("time")

        entropy_t = [compute_entropy(np.outer(ψ, ψ.conj())) for ψ in ψ_t]
        ax[1].plot(times, entropy_t)
        ax[1].set_title("Entropy vs. Time")
        ax[1].set_xlabel("time")
        ax[1].set_ylabel("Entropy")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

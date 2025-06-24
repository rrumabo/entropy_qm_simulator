def compute_purity(rho):
    return np.trace(rho @ rho).real

def compute_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals, 1e-12, 1)
    return -np.sum(evals * np.log(evals))

def compute_localization_width(psi, x):
    ρ = np.abs(psi)**2
    mean = np.sum(ρ * x)
    mean_sq = np.sum(ρ * x**2)
    return np.sqrt(mean_sq - mean**2)

def compute_norm_drift(psi):
    return np.abs(np.sum(np.abs(psi)**2) - 1.0)

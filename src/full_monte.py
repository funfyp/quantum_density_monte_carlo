#!/usr/bin/env python3
"""
Quantum game-sim: density-matrix -> POVM -> probabilities -> entropy & expected payoff
+ Composite scalar F = alpha^{c^2} / ( (sum_lambda) * Psi^3 ) * ( pi_num * (3/5) / Pi_num^2 )

Defaults:
 - 4D Hilbert space, rho = I/4
 - 3-outcome POVM (miss, hit, crit)
 - payoff: 0, 1, 3/5
 - entropy in bits (log2)
 - grid: 100 x 100 (same rho per cell by default; can vary)
Usage:
    python src/full_monte.py --grid 100 --steps 10 --seed 7 --c 2.0 --alpha 0.1
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh
import math

# -------------------------
# Utilities
# -------------------------
def set_seed(seed):
    np.random.seed(seed)

def maximally_mixed(dim=4):
    return np.eye(dim, dtype=complex) / dim

def random_su4():
    # Generate a random Hermitian then exponentiate to get a unitary
    dim = 4
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (A + A.conj().T) / 2
    U = expm(-1j * H)
    # Project to SU(4) by adjusting global phase (det -> 1)
    det = np.linalg.det(U)
    if det == 0:
        return U
    U = U / (det ** (1/4))
    return U

def apply_unitary(rho, U):
    return U @ rho @ U.conj().T

def dephasing_channel(rho, p_dephase=0.1):
    # Simple dephasing mixing off-diagonals with diagonal
    rho = (1 - p_dephase) * rho + p_dephase * np.diag(np.diag(rho))
    # renormalize small numerical drift
    return rho / np.trace(rho)

def is_valid_density(rho, tol=1e-10):
    if rho.shape[0] != rho.shape[1]:
        return False, "not square"
    if not np.allclose(rho, rho.conj().T, atol=1e-8):
        return False, "not Hermitian"
    eigs = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    if eigs.min() < -tol:
        return False, f"negative eigen {eigs.min()}"
    tr = np.trace(rho)
    if not np.isclose(tr, 1.0, atol=1e-8):
        return False, f"trace != 1 ({tr})"
    return True, "ok"

def project_to_physical(rho):
    # Hermitize
    rho = (rho + rho.conj().T) / 2
    vals, vecs = eigh(rho)
    vals_clipped = np.clip(vals, 0, None)
    s = vals_clipped.sum()
    if s == 0:
        return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
    rho_proj = (vecs @ np.diag(vals_clipped) @ vecs.conj().T)
    return rho_proj / np.trace(rho_proj)

def shannon_entropy(p, base=2):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    if base == 2:
        return -np.sum(p * np.log2(p))
    else:
        return -np.sum(p * np.log(p)) / np.log(base)

# -------------------------
# POVM & payoff
# -------------------------
def default_povm():
    # 3-outcome diagonal POVM for 4D Hilbert space
    E_miss = np.diag([1,0,0,0]).astype(complex)
    E_hit  = np.diag([0,1,1,0]).astype(complex)
    E_crit = np.diag([0,0,0,1]).astype(complex)
    return [E_miss, E_hit, E_crit]

def default_payoff():
    return {'miss': 0.0, 'hit': 1.0, 'crit': 3.0/5.0}

def povm_probs(rho, povm):
    probs = np.array([np.real(np.trace(rho @ E)) for E in povm], dtype=float)
    # numerical cleanup
    probs = np.clip(probs, 0.0, 1.0)
    total = probs.sum()
    if total <= 0:
        probs = np.ones(len(povm)) / len(povm)
    else:
        probs = probs / total
    return probs

# -------------------------
# Composite F scalar
# -------------------------
def compute_F(alpha, c, sum_lambda, H_bits=None, Psi_user=None, pi_num=1.0, Pi_num=1.0):
    """
    Compute F = alpha^{c^2} / ( sum_lambda * Psi^3 ) * ( pi_num * (3/5) / Pi_num^2 )
    - alpha: |<0|1>| in [0,1]
    - c: nonnegative float
    - sum_lambda: positive scalar (sum over Omega eigenvalues)
    - H_bits: Shannon entropy (bits). If Psi_user None, Psi = 2^H_bits is used.
    - Psi_user: optional override for Psi (positive)
    - pi_num, Pi_num: positive numeric knobs
    """
    if sum_lambda <= 0:
        raise ValueError("sum_lambda must be > 0")
    # alpha power with 0^0 convention -> 1
    if alpha == 0.0:
        alpha_pow = 1.0 if c == 0.0 else 0.0
    else:
        alpha_pow = math.exp((c**2) * math.log(alpha))
    # choose Psi
    if Psi_user is not None:
        Psi = float(Psi_user)
    else:
        if H_bits is None:
            raise ValueError("H_bits must be provided if Psi_user is not set")
        Psi = 2.0 ** float(H_bits)
    if Psi == 0.0 or Pi_num == 0.0:
        raise ValueError("Psi and Pi_num must be nonzero")
    prefactor = (pi_num * (3.0/5.0)) / (Pi_num**2)
    F = (alpha_pow / (sum_lambda * (Psi**3))) * prefactor
    return F

# -------------------------
# Main grid simulation
# -------------------------
def run_grid_sim(grid_size=(100,100), steps=10, p_dephase=0.1, seed=7,
                 vary_rho=False, energy_levels=None,
                 alpha=0.1, c=2.0, omega=(0,1), Psi_user=None, pi_num=1.0, Pi_num=1.0):
    set_seed(seed)
    dim = 4
    rho0 = maximally_mixed(dim)
    povm = default_povm()
    payoff = default_payoff()
    # arrays to hold scalar outputs per cell
    H_grid = np.zeros(grid_size)
    Epoints_grid = np.zeros(grid_size)
    purity_grid = np.zeros(grid_size)
    eigs_grid = np.zeros(grid_size + (dim,))  # store eigenvalues
    energy_grid = np.zeros(grid_size) if energy_levels is not None else None
    F_grid = np.zeros(grid_size)

    # iterate cells
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # per-cell rho (either same or small random perturbation if vary_rho)
            if not vary_rho:
                rho = rho0.copy()
            else:
                v = np.random.randn(dim) + 1j * np.random.randn(dim)
                v = v / np.linalg.norm(v)
                rho = 0.9 * rho0 + 0.1 * np.outer(v, v.conj())

            # evolve for 'steps' iterations: unitary + dephasing
            for _ in range(steps):
                U = random_su4()
                rho = apply_unitary(rho, U)
                rho = dephasing_channel(rho, p_dephase)
                ok, msg = is_valid_density(rho)
                if not ok:
                    rho = project_to_physical(rho)

            # compute probs, entropy, expected points, purity, eigenvalues
            probs = povm_probs(rho, povm)  # length 3
            H_bits = shannon_entropy(probs, base=2)
            H_grid[i,j] = H_bits
            u_vals = np.array([payoff['miss'], payoff['hit'], payoff['crit']], dtype=float)
            Epoints_grid[i,j] = float(np.dot(u_vals, probs))
            purity_grid[i,j] = float(np.real(np.trace(rho @ rho)))
            eigs = np.real(np.linalg.eigvals(rho))
            eigs = np.sort(eigs)[::-1]
            eigs_grid[i,j,:] = eigs
            if energy_levels is not None:
                Hmat = np.diag(energy_levels)
                energy_grid[i,j] = np.real(np.trace(rho @ Hmat))

            # sum_lambda over provided Omega indices (guard indices)
            idxs = [k for k in omega if 0 <= k < dim]
            if len(idxs) == 0:
                sum_lambda = eigs.sum()
            else:
                sum_lambda = float(np.sum(eigs[idxs]))
            # compute F with Psi = 2^H_bits by default (unless Psi_user provided)
            try:
                F_val = compute_F(alpha=alpha, c=c, sum_lambda=sum_lambda,
                                  H_bits=H_bits, Psi_user=Psi_user,
                                  pi_num=pi_num, Pi_num=Pi_num)
            except Exception as e:
                F_val = 0.0
            F_grid[i,j] = F_val

    # Save figures
    plt.figure(figsize=(14,4))
    plt.subplot(1,4,1)
    plt.title("Shannon Entropy (bits)")
    plt.imshow(H_grid, cmap='viridis')
    plt.colorbar()
    plt.subplot(1,4,2)
    plt.title("Expected Points")
    plt.imshow(Epoints_grid, cmap='plasma')
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.title("Purity")
    plt.imshow(purity_grid, cmap='inferno')
    plt.colorbar()
    plt.subplot(1,4,4)
    plt.title("Composite F")
    plt.imshow(F_grid, cmap='magma')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("figs/grid_outputs_with_F.png", dpi=150)
    plt.close()

    # Eigenvalue figure (e.g., mean top eigenvalue map)
    top_eig = eigs_grid[:,:,0]
    plt.figure()
    plt.title("Top eigenvalue (map)")
    plt.imshow(top_eig, cmap='magma')
    plt.colorbar()
    plt.savefig("figs/top_eig.png", dpi=150)
    plt.close()

    # Save numeric outputs
    np.savez("outputs/sim_data_with_F.npz",
             H_grid=H_grid, Epoints_grid=Epoints_grid, purity_grid=purity_grid,
             eigs_grid=eigs_grid, energy_grid=energy_grid, F_grid=F_grid)
    print("Simulation complete. Figures in figs/, data in outputs/sim_data_with_F.npz")
    return {
        'H': H_grid, 'Epoints': Epoints_grid, 'purity': purity_grid,
        'eigs': eigs_grid, 'energy': energy_grid, 'F': F_grid
    }

# -------------------------
# CLI
# -------------------------
def parse_omega(s):
    # parse comma-separated indices like "0,1" into tuple of ints
    if s is None or s.strip() == "":
        return (0,1)
    parts = s.split(',')
    idxs = []
    for p in parts:
        try:
            idxs.append(int(p.strip()))
        except:
            pass
    return tuple(idxs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=100, help="grid dimension (NxN)")
    parser.add_argument("--steps", type=int, default=10, help="evolution steps")
    parser.add_argument("--pdeph", type=float, default=0.1, help="dephasing prob")
    parser.add_argument("--seed", type=int, default=7, help="rng seed")
    parser.add_argument("--vary-rho", action="store_true", help="add small per-cell rho variation")
    parser.add_argument("--alpha", type=float, default=0.1, help="overlap alpha = |<0|1>| (0..1)")
    parser.add_argument("--c", type=float, default=2.0, help="exponent control c (>=0)")
    parser.add_argument("--omega", type=str, default="0,1", help="comma-separated Omega indices (0-based)")
    parser.add_argument("--Psi", type=float, default=None, help="optional override for Psi (positive)")
    parser.add_argument("--pi-num", type=float, default=1.0, help="numeric pi_num knob")
    parser.add_argument("--Pi-num", type=float, default=1.0, help="numeric Pi_num knob")
    args = parser.parse_args()

    import os
    os.makedirs("figs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    omega = parse_omega(args.omega)
    run_grid_sim(grid_size=(args.grid, args.grid),
                 steps=args.steps,
                 p_dephase=args.pdeph,
                 seed=args.seed,
                 vary_rho=args.vary_rho,
                 alpha=args.alpha,
                 c=args.c,
                 omega=omega,
                 Psi_user=args.Psi,
                 pi_num=args.pi_num,
                 Pi_num=args.Pi_num)

if __name__ == "__main__":
    main()

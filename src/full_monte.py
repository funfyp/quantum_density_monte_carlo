#!/usr/bin/env python3
"""
Quantum Density Matrix Monte Carlo Framework
7-Layer Simulation: 4D Hilbert space + SU(4) + Decoherence + Monte Carlo

Author: Lovely Rhythmic Melody
Date: December 13, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D

# Seed for reproducibility
np.random.seed(42)

# ===== LAYER 1: QUANTUM BACKBONE =====
print("\n=== LAYER 1: QUANTUM BACKBONE ===")
dim = 4
rho = np.eye(dim) / dim  # Maximally mixed state I/4

# POVM measurement operators (projective)
E_miss = np.diag([1, 0, 0, 0])
E_hit = np.diag([0, 1, 1, 0])
E_crit = np.diag([0, 0, 0, 1])

# Verify POVM completeness
assert np.allclose(E_miss + E_hit + E_crit, np.eye(dim))

# Initial probabilities
p_init = np.array([
    np.trace(rho @ E_miss).real,
    np.trace(rho @ E_hit).real,
    np.trace(rho @ E_crit).real
])
print(f"Initial probabilities (miss, hit, crit): {p_init}")
print(f"Expected points (static): {np.dot([0, 1, 0.6], p_init):.4f}")

# ===== LAYER 2: UNITARY DYNAMICS =====
print("\n=== LAYER 2: UNITARY DYNAMICS ===")

def random_su4():
    """Generate random SU(4) unitary via expm(-iH)"""
    H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (H + H.conj().T) / 2  # Make Hermitian
    U = expm(-1j * H)
    return U

# Evolve density matrix 10 steps
for step in range(10):
    U = random_su4()
    rho = U @ rho @ U.conj().T

# Probabilities after evolution
p_evolved = np.array([
    np.trace(rho @ E_miss).real,
    np.trace(rho @ E_hit).real,
    np.trace(rho @ E_crit).real
])
print(f"Evolved probabilities (miss, hit, crit): {p_evolved}")
print(f"Expected points (evolved): {np.dot([0, 1, 0.6], p_evolved):.4f}")

# ===== LAYER 3: DECOHERENCE =====
print("\n=== LAYER 3: DECOHERENCE ===")

# Dephasing channel: diagonal suppression
decoherent_rho = np.diag(np.diag(rho))  # Keep only diagonal
decoherent_rho = decoherent_rho / np.trace(decoherent_rho)  # Renormalize

p_decoherent = np.array([
    np.trace(decoherent_rho @ E_miss).real,
    np.trace(decoherent_rho @ E_hit).real,
    np.trace(decoherent_rho @ E_crit).real
])
print(f"Decoherent probabilities (miss, hit, crit): {p_decoherent}")
print(f"Expected points (decoherent): {np.dot([0, 1, 0.6], p_decoherent):.4f}")

# ===== LAYER 4: RELATIVISTIC ENERGY =====
print("\n=== LAYER 4: RELATIVISTIC ENERGY E=mc² ===")

masses = np.array([1.0, 0.5, 2.5, 1.5])  # Rest masses for basis states
c = 1.0  # Speed of light (natural units)
energies = masses * c**2

H_energy = np.diag(energies)
E_mc2 = np.trace(decoherent_rho @ H_energy).real
print(f"Expected energy <E>: {E_mc2:.4f}")

# ===== LAYER 5: ENTROPY METRICS =====
print("\n=== LAYER 5: ENTROPY METRICS ===")

# Shannon entropy over measurement outcomes
H_shannon = -np.sum(p_decoherent * np.log2(p_decoherent + 1e-12))
H_max_shannon = np.log2(3)  # Max for 3 outcomes
print(f"Shannon entropy (outcomes): {H_shannon:.4f} bits (max: {H_max_shannon:.3f})")

# Von Neumann entropy of state
eigenvalues = np.linalg.eigvalsh(decoherent_rho)
eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Filter numerical zeros
S_von_neumann = -np.sum(eigenvalues * np.log2(eigenvalues))
print(f"Von Neumann entropy S(ρ): {S_von_neumann:.4f} bits")

# ===== LAYER 6: MONTE CARLO SIMULATION =====
print("\n=== LAYER 6: MONTE CARLO SIMULATION ===")
print("100×100 grid, 100 samples/cell = 1,000,000 total")

grid_size = (100, 100)
samples_per_cell = 100
total_samples = grid_size[0] * grid_size[1] * samples_per_cell

# Generate samples from p_decoherent
probs = p_decoherent
outcomes = np.random.choice(3, size=(*grid_size, samples_per_cell), p=probs)

# Calculate statistics per cell
points_map = np.zeros(grid_size)
crit_freq = np.zeros(grid_size)
miss_freq = np.zeros(grid_size)
hit_freq = np.zeros(grid_size)

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        cell_outcomes = outcomes[i, j, :]
        # Points: 0=miss, 1=hit, 0.6=crit
        cell_points = np.where(cell_outcomes == 0, 0,
                              np.where(cell_outcomes == 1, 1, 0.6))
        points_map[i, j] = np.mean(cell_points)
        crit_freq[i, j] = 100 * np.mean(cell_outcomes == 2)
        miss_freq[i, j] = 100 * np.mean(cell_outcomes == 0)
        hit_freq[i, j] = 100 * np.mean(cell_outcomes == 1)

# Global statistics
print(f"\nGlobal statistics across all {total_samples:,} samples:")
print(f"Mean points: {np.mean(points_map):.4f}")
print(f"Crit frequency: {np.mean(crit_freq):.2f}%")
print(f"Miss frequency: {np.mean(miss_freq):.2f}%")
print(f"Hit frequency: {np.mean(hit_freq):.2f}%")
print(f"Theoretical ⟨A⟩ from ρ: {np.dot([0, 1, 0.6], p_decoherent):.4f}")

# ===== LAYER 7: EXPRESSION EVALUATION =====
print("\n=== LAYER 7: EXPRESSION EVALUATION ===")
print("Computing: ⟨0|1⟩^c² / (Σλ_i·Ω) · Ψ³/Π² · π · 3/5")

# Define components
overlap_01 = 0.0  # Orthonormal basis states
c = 4.0
lambda_sum = np.sum(eigenvalues)  # Should be ~1 for valid density matrix
von_neumann_H = S_von_neumann
Pi_squared = np.pi ** 2
Omega = 2 ** von_neumann_H  # Hilbert space volume measure
Psi_cubed = Omega ** 3

# Handle overlap^c² edge case
if overlap_01 == 0:
    overlap_term = 0.0
else:
    overlap_term = overlap_01 ** (c ** 2)

# Full expression
numerator = overlap_term
denominator = lambda_sum * Omega
middle_term = (Psi_cubed / Pi_squared) * np.pi * (3 / 5)

if denominator != 0:
    expression_value = (numerator / denominator) * middle_term
else:
    expression_value = 0.0

print(f"\n⟨0|1⟩ = {overlap_01}")
print(f"c = {c}")
print(f"⟨0|1⟩^(c²) = {overlap_term}")
print(f"Σλ_i = Tr(ρ) = {lambda_sum:.6f}")
print(f"Ω = 2^H(ρ) = {Omega:.6f}")
print(f"Ψ³ = Ω³ = {Psi_cubed:.6f}")
print(f"π·(3/5) = {np.pi * 3/5:.6f}")
print(f"\n**FINAL EXPRESSION VALUE: {expression_value:.6f}**")

# ===== EIGENVALUE DECOMPOSITION =====
print("\n=== EIGENVALUE DECOMPOSITION OF FINAL ρ ===")
eigenvalues_full = np.linalg.eigvalsh(decoherent_rho)
print(f"Eigenvalues (probabilities in eigenbasis): {eigenvalues_full}")
purity = np.trace(decoherent_rho @ decoherent_rho).real
print(f"Purity Tr(ρ²) = {purity:.4f}")

# ===== VISUALIZATION =====
print("\n=== GENERATING VISUALIZATIONS ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Binary grid (10x10 subsample for visibility)
ax = axes[0, 0]
binary_grid = (outcomes[:10, :10, 0] == 2).astype(int)
ax.imshow(binary_grid, cmap='binary', interpolation='nearest')
ax.set_title('Binary Grid (10×10 subsample): 1=Crit, 0=Other')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Dot matrix: brightness = points, size = crit%
ax = axes[0, 1]
X, Y = np.meshgrid(np.arange(100), np.arange(100))
sc = ax.scatter(X.ravel(), Y.ravel(), 
                c=points_map.ravel(), 
                s=crit_freq.ravel() * 5,
                cmap='viridis', alpha=0.6)
ax.set_title('Dot Matrix: Color=Points, Size=Crit%')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(sc, ax=ax, label='Mean Points')

# Histogram of mean points
ax = axes[0, 2]
ax.hist(points_map.ravel(), bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(points_map), color='red', linestyle='--', 
           label=f'Mean: {np.mean(points_map):.4f}')
ax.set_xlabel('Mean Points per Cell')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Cell Mean Points')
ax.legend()
ax.grid(True, alpha=0.3)

# 3D surface: points landscape
ax = axes[1, 0]
ax.remove()
ax = fig.add_subplot(234, projection='3d')
X_grid, Y_grid = np.meshgrid(np.arange(0, 100, 2), np.arange(0, 100, 2))
Z_grid = points_map[::2, ::2]
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='hot', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Mean Points')
ax.set_title('3D Points Landscape')
fig.colorbar(surf, ax=ax, shrink=0.5)

# Crit frequency heatmap
ax = axes[1, 1]
im = ax.imshow(crit_freq, cmap='plasma', aspect='auto')
ax.set_title('Crit Frequency Heatmap (%)')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='Crit %')

# Eigenvalue bar chart
ax = axes[1, 2]
ax.bar(range(len(eigenvalues_full)), eigenvalues_full, 
       color='steelblue', edgecolor='black')
ax.set_xlabel('Eigenvalue Index')
ax.set_ylabel('Eigenvalue (Probability)')
ax.set_title('Density Matrix Eigenvalues')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/full_simulation.png', dpi=150, bbox_inches='tight')
print("Visualization saved: outputs/full_simulation.png")

print("\n=== SIMULATION COMPLETE ===")
print(f"Total runtime layers: 7")
print(f"Final state purity: {purity:.4f}")
print(f"Expression value: {expression_value:.6f}")
print(f"Monte Carlo samples: {total_samples:,}")

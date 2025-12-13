import numpy as np
from src import full_monte as fm


def test_project_and_validate():
    # create an invalid Hermitian with a negative eigenvalue
    A = np.array([[1.0, 2.0], [2.0, -10.0]], dtype=float)
    # embed into 4x4
    rho = np.zeros((4,4), dtype=complex)
    rho[:2,:2] = A
    rho = (rho + rho.conj().T) / 2
    proj = fm.project_to_physical(rho)
    ok, msg = fm.is_valid_density(proj)
    assert ok


def test_povm_probs_normalized():
    rho = fm.maximally_mixed(4)
    povm = fm.default_povm()
    probs = fm.povm_probs(rho, povm)
    assert np.all(probs >= 0)
    assert np.isclose(probs.sum(), 1.0)


def test_compute_F_basic():
    # simple check: alpha=1 -> alpha^{c^2}=1
    F = fm.compute_F(alpha=1.0, c=3.0, sum_lambda=0.5, H_bits=1.0, pi_num=1.0, Pi_num=1.0)
    # with Psi=2^1=2 -> Psi^3=8; prefactor=3/5
    expected = 1.0 / (0.5 * 8.0) * (3.0/5.0)
    assert np.isclose(F, expected)

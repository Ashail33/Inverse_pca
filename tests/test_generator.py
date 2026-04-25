"""Unit tests for the inverse PCA synthetic data generator."""

from __future__ import annotations

import numpy as np
import pytest

from inverse_pca import InversePCAGenerator, make_synthetic_dataset


def test_sample_shapes_and_determinism() -> None:
    gen = InversePCAGenerator(
        n_features=20, n_components=4, random_state=42
    )
    X1, Z1 = gen.sample(50, return_latent=True)
    assert X1.shape == (50, 20)
    assert Z1.shape == (50, 4)

    gen2 = InversePCAGenerator(
        n_features=20, n_components=4, random_state=42
    )
    X2 = gen2.sample(50)
    np.testing.assert_allclose(X1, X2)


def test_basis_is_orthonormal() -> None:
    gen = InversePCAGenerator(n_features=30, n_components=6, random_state=0)
    V = gen.components_
    assert V.shape == (30, 6)
    np.testing.assert_allclose(V.T @ V, np.eye(6), atol=1e-10)


def test_spectrum_decay_matches_request() -> None:
    gen = InversePCAGenerator(
        n_features=10, n_components=4, spectrum="power", spectrum_decay=2.0
    )
    expected = np.array([1.0, 0.25, 1 / 9, 1 / 16])
    np.testing.assert_allclose(gen.explained_variance_, expected)


def test_explicit_spectrum_and_basis() -> None:
    rng = np.random.default_rng(1)
    V, _ = np.linalg.qr(rng.standard_normal((8, 3)))
    variances = np.array([4.0, 1.0, 0.25])
    gen = InversePCAGenerator(
        n_features=8,
        n_components=3,
        spectrum=variances,
        basis=V,
        random_state=1,
    )
    np.testing.assert_allclose(gen.components_, V)
    np.testing.assert_allclose(gen.explained_variance_, variances)


def test_population_covariance_recovered_in_large_sample() -> None:
    gen = InversePCAGenerator(
        n_features=15,
        n_components=5,
        spectrum="exponential",
        spectrum_decay=0.5,
        noise_std=0.1,
        random_state=7,
    )
    X = gen.sample(20_000)
    emp_cov = np.cov(X, rowvar=False)
    pop_cov = gen.covariance()
    err = np.linalg.norm(emp_cov - pop_cov) / np.linalg.norm(pop_cov)
    assert err < 0.05


def test_transform_inverse_transform_round_trip() -> None:
    gen = InversePCAGenerator(
        n_features=12, n_components=4, spectrum="uniform", random_state=3
    )
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((25, 4))
    Z_hat = gen.inverse_transform(gen.transform(Z))
    np.testing.assert_allclose(Z, Z_hat, atol=1e-10)


def test_mean_offset_is_applied() -> None:
    mean = np.linspace(-1, 1, 6)
    gen = InversePCAGenerator(
        n_features=6,
        n_components=2,
        spectrum="uniform",
        mean=mean,
        random_state=2,
    )
    X = gen.sample(5_000)
    np.testing.assert_allclose(X.mean(axis=0), mean, atol=0.1)


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        InversePCAGenerator(n_features=4, n_components=10)
    with pytest.raises(ValueError):
        InversePCAGenerator(n_features=4, n_components=2, noise_std=-1)
    with pytest.raises(ValueError):
        InversePCAGenerator(
            n_features=4, n_components=2, spectrum=np.array([1.0, -1.0])
        )

    bad_basis = np.ones((5, 2))  # not orthonormal
    with pytest.raises(ValueError):
        InversePCAGenerator(n_features=5, n_components=2, basis=bad_basis)


def test_custom_latent_callable() -> None:
    def laplace_like(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
        return rng.laplace(scale=1 / np.sqrt(2.0), size=(n, k))

    gen = InversePCAGenerator(
        n_features=10,
        n_components=3,
        latent_dist=laplace_like,
        spectrum="uniform",
        random_state=11,
    )
    X = gen.sample(2_000)
    # Variance along principal axes should be ~1 because spectrum is uniform.
    proj = (X - X.mean(axis=0)) @ gen.components_
    np.testing.assert_allclose(proj.var(axis=0), np.ones(3), atol=0.15)


def test_make_synthetic_dataset_helper() -> None:
    X, Z, gen = make_synthetic_dataset(
        n_samples=100, n_features=8, n_components=3, random_state=5
    )
    assert X.shape == (100, 8)
    assert Z.shape == (100, 3)
    assert isinstance(gen, InversePCAGenerator)

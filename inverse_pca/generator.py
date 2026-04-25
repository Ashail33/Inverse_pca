"""Statistical inverse PCA generator.

Forward PCA decomposes a data matrix as

    X - mu = Z @ diag(sqrt(lambda)) @ V.T + E

where V (d, k) holds orthonormal principal axes, lambda (k,) holds the
component variances, Z (n, k) holds standardised latent scores and E is
residual noise. The *statistical inverse* runs this generative model in the
opposite direction: choose mu, V, lambda and a latent distribution, then draw
samples X from the implied population. This module exposes that construction
as :class:`InversePCAGenerator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]
SpectrumSpec = Union[str, ArrayLike, Callable[[int], np.ndarray]]
LatentSpec = Union[str, Callable[[int, int, np.random.Generator], np.ndarray]]


def _build_spectrum(spec: SpectrumSpec, k: int, decay: float) -> np.ndarray:
    """Return a length-k vector of non-negative component variances."""
    if callable(spec):
        values = np.asarray(spec(k), dtype=float)
    elif isinstance(spec, str):
        idx = np.arange(1, k + 1, dtype=float)
        if spec == "power":
            values = idx ** (-decay)
        elif spec == "exponential":
            values = np.exp(-decay * (idx - 1))
        elif spec == "linear":
            values = np.clip(1.0 - (idx - 1) / k, a_min=1e-12, a_max=None)
        elif spec == "uniform":
            values = np.ones(k)
        else:
            raise ValueError(f"Unknown spectrum '{spec}'.")
    else:
        values = np.asarray(spec, dtype=float)
        if values.shape != (k,):
            raise ValueError(
                f"Explicit spectrum has shape {values.shape}, expected ({k},)."
            )

    if np.any(values < 0):
        raise ValueError("Spectrum entries (variances) must be non-negative.")
    return values


def _random_orthonormal_basis(
    n_features: int, n_components: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw a uniformly random (d, k) matrix with orthonormal columns."""
    if n_components > n_features:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed n_features ({n_features})."
        )
    g = rng.standard_normal(size=(n_features, n_components))
    q, r = np.linalg.qr(g)
    # Fix sign ambiguity so the basis is reproducible across LAPACK versions.
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def _draw_latent(
    spec: LatentSpec,
    n_samples: int,
    n_components: int,
    rng: np.random.Generator,
    *,
    df: float = 5.0,
) -> np.ndarray:
    """Draw an (n, k) matrix of standardised latent scores."""
    if callable(spec):
        z = np.asarray(spec(n_samples, n_components, rng), dtype=float)
        if z.shape != (n_samples, n_components):
            raise ValueError(
                f"Latent sampler returned shape {z.shape}, expected "
                f"({n_samples}, {n_components})."
            )
        return z
    if spec == "gaussian":
        return rng.standard_normal(size=(n_samples, n_components))
    if spec == "uniform":
        # Scaled so that variance is 1 along each axis.
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n_samples, n_components))
    if spec == "laplace":
        return rng.laplace(scale=1.0 / np.sqrt(2.0), size=(n_samples, n_components))
    if spec == "t":
        if df <= 2:
            raise ValueError("Student-t latent requires df > 2 for finite variance.")
        raw = rng.standard_t(df, size=(n_samples, n_components))
        return raw * np.sqrt((df - 2.0) / df)
    raise ValueError(f"Unknown latent distribution '{spec}'.")


@dataclass
class InversePCAGenerator:
    """Generate high-dimensional samples from a prescribed PCA structure.

    Parameters
    ----------
    n_features:
        Output dimensionality d.
    n_components:
        Latent dimensionality k. Must satisfy ``k <= d``.
    spectrum:
        How component variances ``lambda`` are chosen. Either one of the
        strings ``"power"``, ``"exponential"``, ``"linear"``, ``"uniform"``,
        an explicit length-k array, or a callable ``f(k) -> array``.
    spectrum_decay:
        Decay parameter used by the ``"power"`` and ``"exponential"`` profiles.
    noise_std:
        Standard deviation of additive isotropic Gaussian noise applied in
        the ambient space (the residual term ``E``).
    mean:
        Optional length-d mean vector ``mu``. Defaults to zero.
    basis:
        Optional (d, k) matrix of orthonormal columns to use as ``V``. If
        omitted, a random orthonormal basis is drawn.
    latent_dist:
        Distribution used to draw standardised latent scores. One of
        ``"gaussian"``, ``"uniform"``, ``"laplace"``, ``"t"``, or a callable
        ``f(n, k, rng) -> array``.
    df:
        Degrees of freedom for the Student-t latent (ignored otherwise).
    random_state:
        Seed or :class:`numpy.random.Generator` used for both basis
        construction and sampling.
    """

    n_features: int
    n_components: int
    spectrum: SpectrumSpec = "power"
    spectrum_decay: float = 1.0
    noise_std: float = 0.0
    mean: Optional[ArrayLike] = None
    basis: Optional[np.ndarray] = None
    latent_dist: LatentSpec = "gaussian"
    df: float = 5.0
    random_state: Union[int, np.random.Generator, None] = None

    def __post_init__(self) -> None:
        if self.n_features <= 0 or self.n_components <= 0:
            raise ValueError("n_features and n_components must be positive.")
        if self.n_components > self.n_features:
            raise ValueError(
                "n_components must not exceed n_features for a valid PCA basis."
            )
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative.")

        self._rng = np.random.default_rng(self.random_state)

        if self.mean is None:
            self._mean = np.zeros(self.n_features)
        else:
            self._mean = np.asarray(self.mean, dtype=float).reshape(-1)
            if self._mean.shape != (self.n_features,):
                raise ValueError(
                    f"mean has shape {self._mean.shape}, expected ({self.n_features},)."
                )

        if self.basis is None:
            self._basis = _random_orthonormal_basis(
                self.n_features, self.n_components, self._rng
            )
        else:
            V = np.asarray(self.basis, dtype=float)
            if V.shape != (self.n_features, self.n_components):
                raise ValueError(
                    f"basis has shape {V.shape}, expected "
                    f"({self.n_features}, {self.n_components})."
                )
            gram = V.T @ V
            if not np.allclose(gram, np.eye(self.n_components), atol=1e-6):
                raise ValueError("Provided basis must have orthonormal columns.")
            self._basis = V

        self._variances = _build_spectrum(
            self.spectrum, self.n_components, self.spectrum_decay
        )
        self._scales = np.sqrt(self._variances)

    # ------------------------------------------------------------------ API

    @property
    def components_(self) -> np.ndarray:
        """Orthonormal principal axes V with shape (d, k)."""
        return self._basis

    @property
    def explained_variance_(self) -> np.ndarray:
        """Component variances lambda with shape (k,)."""
        return self._variances

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Fraction of total signal variance carried by each component."""
        total = self._variances.sum()
        if total == 0:
            return np.zeros_like(self._variances)
        return self._variances / total

    @property
    def mean_(self) -> np.ndarray:
        return self._mean

    def covariance(self) -> np.ndarray:
        """Population covariance ``V diag(lambda) V.T + sigma^2 I``."""
        signal = (self._basis * self._variances) @ self._basis.T
        if self.noise_std > 0:
            signal = signal + (self.noise_std ** 2) * np.eye(self.n_features)
        return signal

    def sample(
        self, n_samples: int, *, return_latent: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Draw ``n_samples`` synthetic observations.

        Returns ``X`` with shape ``(n_samples, n_features)`` and, if
        ``return_latent`` is true, also the latent scores ``Z`` with shape
        ``(n_samples, n_components)``.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        z = _draw_latent(
            self.latent_dist, n_samples, self.n_components, self._rng, df=self.df
        )
        x = self.transform(z)
        if self.noise_std > 0:
            x = x + self._rng.normal(
                scale=self.noise_std, size=(n_samples, self.n_features)
            )
        if return_latent:
            return x, z
        return x

    def transform(self, latent: ArrayLike) -> np.ndarray:
        """Map latent scores Z (n, k) into ambient space (no residual noise)."""
        z = np.atleast_2d(np.asarray(latent, dtype=float))
        if z.shape[1] != self.n_components:
            raise ValueError(
                f"latent has {z.shape[1]} columns, expected {self.n_components}."
            )
        return self._mean + (z * self._scales) @ self._basis.T

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Project ambient samples back to standardised latent scores."""
        x = np.atleast_2d(np.asarray(X, dtype=float))
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"X has {x.shape[1]} columns, expected {self.n_features}."
            )
        scores = (x - self._mean) @ self._basis
        safe = np.where(self._scales > 0, self._scales, 1.0)
        return np.where(self._scales > 0, scores / safe, 0.0)


def make_synthetic_dataset(
    n_samples: int,
    n_features: int,
    n_components: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, InversePCAGenerator]:
    """Convenience wrapper: build a generator and draw a labelled dataset.

    Returns ``(X, Z, generator)`` where ``X`` is the high-dimensional data,
    ``Z`` is the latent scores used to produce it, and ``generator`` is the
    fitted :class:`InversePCAGenerator` (useful for inspecting the spectrum
    or sampling further batches).
    """
    gen = InversePCAGenerator(
        n_features=n_features, n_components=n_components, **kwargs
    )
    X, Z = gen.sample(n_samples, return_latent=True)
    return X, Z, gen

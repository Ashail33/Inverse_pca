"""Test whether 2-D cluster structure survives an inverse-PCA embedding.

For five different cluster topologies (blobs, anisotropic, concentric
circles, two moons, spiral) we:

    1. draw the latent Z in 2-D using a hand-built generator;
    2. push it into d=50 dimensions via InversePCAGenerator.transform,
       optionally adding isotropic noise;
    3. re-fit PCA on the high-dim sample and project onto its top two
       components;
    4. Procrustes-align the recovery to the original so rotation/sign
       ambiguity does not obscure the comparison;
    5. plot original vs. recovered side by side.

Run as:

    PYTHONPATH=. python examples/plot_cluster_preservation.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from inverse_pca import InversePCAGenerator

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2-D cluster patterns. Each returns (Z, labels) where Z has shape (n, 2).
# ---------------------------------------------------------------------------
def blobs(n: int, rng: np.random.Generator):
    centres = np.array([[2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]])
    labels = rng.integers(0, 3, size=n)
    Z = centres[labels] + rng.standard_normal((n, 2)) * 0.45
    return Z, labels


def anisotropic(n: int, rng: np.random.Generator):
    labels = rng.integers(0, 2, size=n)
    Z = np.empty((n, 2))
    # Cluster 0: elongated along x-axis
    m0 = labels == 0
    Z[m0] = (np.array([2.5, 0.0])
             + rng.standard_normal((m0.sum(), 2)) * np.array([1.4, 0.18]))
    # Cluster 1: elongated along the (1, 1) direction
    m1 = ~m0
    raw = rng.standard_normal((m1.sum(), 2)) * np.array([1.4, 0.18])
    rot = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                    [np.sin(np.pi / 4),  np.cos(np.pi / 4)]])
    Z[m1] = np.array([-2.0, 0.0]) + raw @ rot.T
    return Z, labels


def concentric_circles(n: int, rng: np.random.Generator):
    labels = rng.integers(0, 2, size=n)
    radii = np.where(labels == 0, 1.0, 2.6)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    Z = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    Z = Z + rng.standard_normal((n, 2)) * 0.08
    return Z, labels


def two_moons(n: int, rng: np.random.Generator):
    labels = rng.integers(0, 2, size=n)
    theta = rng.uniform(0, np.pi, size=n)
    x = np.where(labels == 0, np.cos(theta), 1.0 - np.cos(theta))
    y = np.where(labels == 0, np.sin(theta), 0.5 - np.sin(theta))
    Z = np.column_stack([x, y]) + rng.standard_normal((n, 2)) * 0.06
    return Z, labels


def spiral(n: int, rng: np.random.Generator):
    t = np.sqrt(rng.uniform(0.05, 1.0, size=n)) * 3.5 * np.pi
    Z = 0.32 * np.column_stack([t * np.cos(t), t * np.sin(t)])
    Z = Z + rng.standard_normal((n, 2)) * 0.06
    # Colour by arc length so we can see whether the spiral order survives.
    labels = (t / np.pi).astype(int)
    return Z, labels


PATTERNS = [
    ("blobs",                blobs),
    ("anisotropic",          anisotropic),
    ("concentric circles",   concentric_circles),
    ("two moons",            two_moons),
    ("spiral",               spiral),
]


# ---------------------------------------------------------------------------
# Embedding + recovery helpers.
# ---------------------------------------------------------------------------
def standardise(Z: np.ndarray) -> np.ndarray:
    return (Z - Z.mean(axis=0)) / Z.std(axis=0)


def embed_and_recover(
    Z: np.ndarray, n_features: int, noise_std: float, random_state: int
):
    """Push Z (n, 2) into n_features-dim and project back via PCA."""
    Zs = standardise(Z)
    gen = InversePCAGenerator(
        n_features=n_features, n_components=2,
        spectrum=np.array([1.0, 1.0]),
        noise_std=noise_std, random_state=random_state,
    )
    rng = np.random.default_rng(random_state + 1)
    X = gen.transform(Zs)
    if noise_std > 0:
        X = X + rng.normal(scale=noise_std, size=X.shape)

    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    top2 = eigvecs[:, order[:2]]
    Z_hat = Xc @ top2
    return Zs, _procrustes_align(Z_hat, Zs)


def _procrustes_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rotate / reflect A to best match B (no scaling)."""
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return A @ R


# ---------------------------------------------------------------------------
# Main figure: original vs. recovered for each pattern, two noise levels.
# ---------------------------------------------------------------------------
def main() -> None:
    n = 1_500
    d = 50
    noise_low = 0.05
    noise_high = 0.40
    rng_master = np.random.default_rng(0)

    fig, axes = plt.subplots(3, len(PATTERNS),
                             figsize=(3.0 * len(PATTERNS), 8.4))
    cmap_continuous = plt.get_cmap("viridis")

    for col, (name, sampler) in enumerate(PATTERNS):
        rng = np.random.default_rng(rng_master.integers(0, 1 << 32))
        Z, labels = sampler(n, rng)
        Zs, Z_low = embed_and_recover(Z, d, noise_low, random_state=col)
        _,  Z_high = embed_and_recover(Z, d, noise_high, random_state=col)

        if name == "spiral":
            colours = cmap_continuous(labels / labels.max())
            scatter_kw = dict(c=colours, s=4, alpha=0.7)
        else:
            colours = labels
            scatter_kw = dict(c=colours, cmap="tab10", s=5, alpha=0.6)

        axes[0, col].scatter(Zs[:, 0], Zs[:, 1], **scatter_kw)
        axes[1, col].scatter(Z_low[:, 0], Z_low[:, 1], **scatter_kw)
        axes[2, col].scatter(Z_high[:, 0], Z_high[:, 1], **scatter_kw)

        axes[0, col].set_title(name)

    row_labels = [
        "original 2-D latent",
        f"top-2 PCA of d={d}\n(noise std = {noise_low})",
        f"top-2 PCA of d={d}\n(noise std = {noise_high})",
    ]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=10)

    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Do 2-D cluster topologies survive a linear inverse-PCA "
        f"embedding into d={d}?",
        y=1.0, fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "06_cluster_preservation.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

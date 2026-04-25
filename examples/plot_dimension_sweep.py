"""Sweep ambient dimensionality d and sample size n.

Pushes the same 2-D two-moons latent through inverse PCA at every
combination of d in {20, 100, 500, 2000} and n in {50, 200, 1000, 5000},
then recovers via top-2 PCA. Shows the Marchenko-Pastur / BBP regime
empirically: when d/n is small the recovery is exact, when d/n grows the
sample covariance becomes noisy and the principal axes drift.

Run as:

    PYTHONPATH=. python examples/plot_dimension_sweep.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from inverse_pca import InversePCAGenerator

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

NOISE_STD = 0.4
D_VALUES = [20, 100, 500, 2000]
N_VALUES = [50, 200, 1000, 5000]
RANDOM_STATE = 0


def two_moons(n: int, rng: np.random.Generator):
    labels = rng.integers(0, 2, size=n)
    theta = rng.uniform(0, np.pi, size=n)
    x = np.where(labels == 0, np.cos(theta), 1.0 - np.cos(theta))
    y = np.where(labels == 0, np.sin(theta), 0.5 - np.sin(theta))
    Z = np.column_stack([x, y]) + rng.standard_normal((n, 2)) * 0.06
    return Z, labels


def standardise(Z: np.ndarray) -> np.ndarray:
    return (Z - Z.mean(axis=0)) / Z.std(axis=0)


def procrustes_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M)
    return A @ (U @ Vt)


def alignment_score(Z_recovered: np.ndarray, Z_truth: np.ndarray) -> float:
    """Pearson-style alignment after Procrustes: 1 = perfect, 0 = random."""
    a = Z_recovered.flatten()
    b = Z_truth.flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_and_recover(Z: np.ndarray, d: int, seed: int, noise_std: float):
    Zs = standardise(Z)
    gen = InversePCAGenerator(
        n_features=d, n_components=2,
        spectrum=np.array([1.0, 1.0]),
        noise_std=noise_std, random_state=seed,
    )
    rng = np.random.default_rng(seed + 7)
    X = gen.transform(Zs) + rng.normal(scale=noise_std, size=(Zs.shape[0], d))
    Xc = X - X.mean(axis=0)
    # Use SVD of the data (works fine when n < d, unlike eigh of the
    # rank-deficient d x d sample covariance).
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    top2 = Vt[:2].T
    Z_hat = Xc @ top2
    return Zs, procrustes_align(Z_hat, Zs)


def grid_figure() -> str:
    rng_master = np.random.default_rng(RANDOM_STATE)
    fig, axes = plt.subplots(
        len(D_VALUES), len(N_VALUES),
        figsize=(2.7 * len(N_VALUES), 2.7 * len(D_VALUES)),
    )
    for r, d in enumerate(D_VALUES):
        for c, n in enumerate(N_VALUES):
            seed = int(rng_master.integers(0, 1 << 30))
            rng = np.random.default_rng(seed)
            Z, labels = two_moons(n, rng)
            Zs, Z_hat = embed_and_recover(Z, d, seed=seed,
                                          noise_std=NOISE_STD)
            score = alignment_score(Z_hat, Zs)
            ratio = d / n

            ax = axes[r, c]
            ax.scatter(Z_hat[:, 0], Z_hat[:, 1],
                       c=labels, cmap="tab10", s=6, alpha=0.6)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(True, alpha=0.25)
            ax.set_title(
                f"d/n = {ratio:.2g}\nalign = {score:.2f}",
                fontsize=9,
            )
    for c, n in enumerate(N_VALUES):
        axes[0, c].set_xlabel(f"n = {n}", fontsize=11)
        axes[0, c].xaxis.set_label_position("top")
    for r, d in enumerate(D_VALUES):
        axes[r, 0].set_ylabel(f"d = {d}", fontsize=11)
    fig.suptitle(
        f"Two-moons recovery vs. (d, n) — fixed noise std = {NOISE_STD}",
        y=1.0, fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "07_dimension_sweep.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def transition_figure() -> str:
    """Alignment score vs. d/n for several noise levels (BBP-style)."""
    n_fixed = 400
    d_grid = np.unique(np.round(
        np.logspace(np.log10(5), np.log10(20_000), 25)
    ).astype(int))
    sigmas = [0.1, 0.3, 0.5, 0.7]
    n_repeats = 5

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for sigma, colour in zip(sigmas, ["C0", "C1", "C2", "C3"]):
        scores = np.zeros(len(d_grid))
        for i, d in enumerate(d_grid):
            seeds = np.arange(n_repeats) + int(d)
            vals = []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                Z, _ = two_moons(n_fixed, rng)
                Zs, Z_hat = embed_and_recover(
                    Z, int(d), seed=int(s), noise_std=sigma
                )
                vals.append(alignment_score(Z_hat, Zs))
            scores[i] = np.mean(vals)
        ax.semilogx(d_grid / n_fixed, scores, "o-",
                    label=f"$\\sigma = {sigma}$", color=colour, markersize=4)

    ax.axhline(1.0, ls="--", color="grey", alpha=0.6, lw=0.8)
    ax.set_xlabel("d / n  (log scale)")
    ax.set_ylabel("alignment with true latent")
    ax.set_title(
        f"PCA recovery quality vs. d/n  (n = {n_fixed}, k = 2, "
        f"averaged over {n_repeats} seeds)"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "08_dn_transition.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print(f"wrote {grid_figure()}")
    print(f"wrote {transition_figure()}")


if __name__ == "__main__":
    main()

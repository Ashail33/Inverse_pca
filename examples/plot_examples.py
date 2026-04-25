"""Generate the figures shown in the README.

Each function builds one or more InversePCAGenerator instances, draws
samples, and saves a PNG into examples/figures/. Run as:

    PYTHONPATH=. python examples/plot_examples.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from inverse_pca import InversePCAGenerator

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Prescribed spectrum vs. spectrum recovered by re-fitting PCA on samples.
# ---------------------------------------------------------------------------
def spectrum_recovery() -> str:
    d, k, n = 200, 10, 4_000
    gen = InversePCAGenerator(
        n_features=d, n_components=k,
        spectrum="power", spectrum_decay=1.5,
        noise_std=0.05, random_state=0,
    )
    X = gen.sample(n)
    cov = np.cov(X, rowvar=False)
    emp = np.linalg.eigvalsh(cov)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    idx = np.arange(1, d + 1)
    ax.semilogy(idx[:30], emp[:30], "o-", label="empirical (sample PCA)",
                color="C0", markersize=5)
    ax.semilogy(np.arange(1, k + 1), gen.explained_variance_, "s",
                label="prescribed signal $\\lambda_i$",
                color="C3", markersize=8)
    ax.axhline(gen.noise_std ** 2, ls="--", color="grey",
               label=f"noise floor $\\sigma^2 = {gen.noise_std**2:.4f}$")
    ax.set_xlabel("component index")
    ax.set_ylabel("variance (log scale)")
    ax.set_title(f"Spectrum recovery: d={d}, k={k}, n={n}, "
                 r"$\lambda_i = i^{-1.5}$")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, "01_spectrum_recovery.png")


# ---------------------------------------------------------------------------
# 2. The four parametric spectrum profiles side by side.
# ---------------------------------------------------------------------------
def spectrum_profiles() -> str:
    k = 12
    fig, ax = plt.subplots(figsize=(7, 4.2))
    profiles = [
        ("power", 1.0, "C0"),
        ("power", 2.0, "C1"),
        ("exponential", 0.5, "C2"),
        ("linear", None, "C3"),
        ("uniform", None, "C4"),
    ]
    for spec, decay, colour in profiles:
        kwargs = dict(spectrum=spec)
        if decay is not None:
            kwargs["spectrum_decay"] = decay
        gen = InversePCAGenerator(n_features=k, n_components=k, **kwargs)
        label = spec if decay is None else f"{spec} (decay={decay})"
        ax.plot(np.arange(1, k + 1), gen.explained_variance_, "o-",
                label=label, color=colour)
    ax.set_yscale("log")
    ax.set_xlabel("component index")
    ax.set_ylabel("$\\lambda_i$ (log)")
    ax.set_title("Built-in spectrum profiles")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, "02_spectrum_profiles.png")


# ---------------------------------------------------------------------------
# 3. Effect of latent distribution on the shape of the cloud (2-d slice).
# ---------------------------------------------------------------------------
def latent_distribution_shapes() -> str:
    distributions = ["gaussian", "uniform", "laplace", "t"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)
    for ax, name in zip(axes, distributions):
        kwargs = dict(
            n_features=2, n_components=2,
            spectrum=np.array([1.0, 1.0]),  # isotropic so shape is the latent
            latent_dist=name, random_state=0,
        )
        if name == "t":
            kwargs["df"] = 5.0
        gen = InversePCAGenerator(**kwargs)
        X = gen.sample(4_000)
        ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.35, color="C0")
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("$x_2$")
    for ax in axes:
        ax.set_xlabel("$x_1$")
    fig.suptitle("Latent distribution shapes (isotropic spectrum, "
                 "random orthonormal basis)", y=1.02)
    return _save(fig, "03_latent_distributions.png")


# ---------------------------------------------------------------------------
# 4. Clustered latent space pushed into 50 dims, viewed in latent + PCA space.
# ---------------------------------------------------------------------------
def clustered_latent() -> str:
    d, k, n = 50, 4, 1_500

    def three_clusters(n_, k_, rng):
        centres = np.array([
            [ 2.0,  0.0, 0.0, 0.0],
            [-1.0,  1.7, 0.0, 0.0],
            [-1.0, -1.7, 0.0, 0.0],
        ])[:, :k_]
        labels = rng.integers(0, 3, size=n_)
        return centres[labels] + rng.standard_normal((n_, k_)) * 0.35, labels

    rng = np.random.default_rng(0)
    Z, labels = three_clusters(n, k, rng)

    gen = InversePCAGenerator(
        n_features=d, n_components=k,
        spectrum="power", spectrum_decay=1.0,
        noise_std=0.1, random_state=0,
    )
    X = gen.transform(Z) + rng.normal(scale=gen.noise_std, size=(n, d))

    # Re-fit PCA on the high-dim sample for visualisation.
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    top2 = eigvecs[:, order[:2]]
    X_proj = Xc @ top2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colours = ["C0", "C1", "C2"]
    for c in range(3):
        m = labels == c
        axes[0].scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6,
                        color=colours[c], label=f"cluster {c}")
        axes[1].scatter(X_proj[m, 0], X_proj[m, 1], s=8, alpha=0.6,
                        color=colours[c], label=f"cluster {c}")
    axes[0].set_title(f"Latent space ($k={k}$): first two latent dims")
    axes[1].set_title(f"Top-2 PCA of synthetic high-dim data ($d={d}$)")
    for ax in axes:
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.suptitle("Cluster structure survives the inverse PCA embedding",
                 y=1.02)
    return _save(fig, "04_clustered_latent.png")


# ---------------------------------------------------------------------------
# 5. Effect of noise level on the empirical spectrum (signal vs. bulk).
# ---------------------------------------------------------------------------
def noise_floor_sweep() -> str:
    d, k, n = 100, 5, 3_000
    sigmas = [0.0, 0.1, 0.3, 0.6]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for sigma, colour in zip(sigmas, ["C0", "C1", "C2", "C3"]):
        gen = InversePCAGenerator(
            n_features=d, n_components=k,
            spectrum=np.array([4.0, 2.0, 1.0, 0.5, 0.25]),
            noise_std=sigma, random_state=0,
        )
        X = gen.sample(n)
        emp = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
        ax.semilogy(np.arange(1, 31), emp[:30], "o-",
                    label=f"$\\sigma={sigma}$", color=colour, markersize=4)
    ax.set_xlabel("component index")
    ax.set_ylabel("empirical eigenvalue (log)")
    ax.set_title(f"Noise floor: 5 signal eigenvalues "
                 f"$+\\,\\sigma^2$ bulk on {d-k} axes")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, "05_noise_floor.png")


def main() -> None:
    paths = [
        spectrum_recovery(),
        spectrum_profiles(),
        latent_distribution_shapes(),
        clustered_latent(),
        noise_floor_sweep(),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

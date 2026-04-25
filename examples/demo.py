"""Generate a 200-d synthetic dataset from a 5-d latent space and verify
that re-fitting PCA recovers the prescribed spectrum.
"""

import numpy as np

from inverse_pca import InversePCAGenerator


def main() -> None:
    gen = InversePCAGenerator(
        n_features=200,
        n_components=5,
        spectrum="power",
        spectrum_decay=1.5,
        noise_std=0.05,
        latent_dist="gaussian",
        random_state=0,
    )

    X, Z = gen.sample(n_samples=2_000, return_latent=True)
    print(f"X shape: {X.shape}, Z shape: {Z.shape}")
    print(f"Prescribed variances: {np.round(gen.explained_variance_, 4)}")

    # Empirically refit PCA on the synthetic sample.
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    print(f"Top-{gen.n_components} empirical eigenvalues: "
          f"{np.round(eigvals[:gen.n_components], 4)}")

    # Round-trip check: latent -> ambient -> latent (no noise).
    Z_hat = gen.inverse_transform(gen.transform(Z))
    print(f"Latent round-trip max error: {np.max(np.abs(Z - Z_hat)):.2e}")


if __name__ == "__main__":
    main()

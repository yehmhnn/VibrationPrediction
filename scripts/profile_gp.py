import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def profile_gp(n):
    x = torch.randn(n, 1, device="cpu")
    t = torch.arange(n, dtype=torch.float64, device="cpu")
    pairwise_sq = (t.unsqueeze(0) - t.unsqueeze(1)) ** 2
    sigma = 1.0
    lengthscale = 1.0
    start = time.perf_counter()
    K = sigma**2 * torch.exp(-0.5 * pairwise_sq / (lengthscale**2))
    build_time = time.perf_counter() - start

    start = time.perf_counter()
    torch.linalg.cholesky(K + 1e-6 * torch.eye(n, device="cpu"))
    chol_time = time.perf_counter() - start
    return build_time, chol_time


def main():
    parser = argparse.ArgumentParser(description="Profile GP prior complexity")
    parser.add_argument(
        "--output", type=Path, default=Path("figures/gp_complexity_profile.png")
    )
    args = parser.parse_args()

    sizes = [64, 128, 256, 512, 1024]
    results = []
    for n in sizes:
        print(f"Profiling N={n}")
        build_time, chol_time = profile_gp(n)
        results.append((n, build_time, chol_time))

    sizes = np.array([r[0] for r in results])
    build_times = np.array([r[1] for r in results])
    chol_times = np.array([r[2] for r in results])

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, build_times, marker="o", label="Covariance build")
    plt.plot(sizes, chol_times, marker="o", label="Cholesky")
    plt.plot(
        sizes,
        sizes**3 * (build_times[0] / (sizes[0] ** 3)),
        linestyle="--",
        label="$O(N^3)$ reference",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Sequence length N")
    plt.ylabel("Time [s]")
    plt.title("GP prior complexity profile")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Saved GP complexity plot to {args.output}")


if __name__ == "__main__":
    main()

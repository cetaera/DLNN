from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIZES = (2, 4, 16)
ENSEMBLE_SIZE = 10_000
SEED = 20261029
HISTOGRAM_LIMIT = 3.5
HISTOGRAM_BINS = 60


def generate_symmetric_ensemble(rng, size, count, distribution):
    """Generate X + X.T for the requested element distribution."""
    if distribution == "gaussian":
        raw = rng.normal(0.0, 1.0, size=(count, size, size))
    elif distribution == "rademacher":
        raw = 2.0 * rng.integers(0, 2, size=(count, size, size)) - 1.0
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    return raw + np.swapaxes(raw, 1, 2)


def central_spacing(matrices):
    """Return one fixed central eigenvalue spacing for every matrix."""
    eigenvalues = np.linalg.eigvalsh(matrices)
    left_index = matrices.shape[1] // 2 - 1
    return eigenvalues[:, left_index + 1] - eigenvalues[:, left_index]


def normalize_to_unit_mean(values):
    mean = float(np.mean(values))
    if not np.isfinite(mean) or mean <= 0.0:
        raise ArithmeticError("the mean eigenvalue spacing must be positive")
    return values / mean, mean


def wigner_density(spacing):
    return 0.5 * np.pi * spacing * np.exp(-0.25 * np.pi * spacing**2)


def wigner_cdf(spacing):
    return 1.0 - np.exp(-0.25 * np.pi * spacing**2)


def kolmogorov_smirnov_distance(values):
    ordered = np.sort(values)
    count = ordered.size
    theoretical = wigner_cdf(ordered)
    upper = np.arange(1, count + 1) / count - theoretical
    lower = theoretical - np.arange(count) / count
    return float(max(np.max(upper), np.max(lower)))


def analyze_ensemble(rng, distribution):
    results = []
    for size in SIZES:
        matrices = generate_symmetric_ensemble(
            rng, size, ENSEMBLE_SIZE, distribution
        )
        gaps = central_spacing(matrices)
        normalized, raw_mean = normalize_to_unit_mean(gaps)
        results.append(
            {
                "size": size,
                "left_index": size // 2 - 1,
                "normalized_gaps": normalized,
                "raw_mean": raw_mean,
                "ks_distance": kolmogorov_smirnov_distance(normalized),
                "near_zero_fraction": float(np.mean(normalized < 0.1)),
                "zero_fraction": float(np.mean(normalized < 1e-12)),
            }
        )
    return results


def plot_results(results, title, output_path):
    figure, axes = plt.subplots(1, len(results), figsize=(15, 4.5), sharey=True)
    edges = np.linspace(0.0, HISTOGRAM_LIMIT, HISTOGRAM_BINS + 1)
    bin_width = edges[1] - edges[0]
    curve_x = np.linspace(0.0, HISTOGRAM_LIMIT, 500)

    for axis, result in zip(axes, results):
        gaps = result["normalized_gaps"]
        weights = np.full(gaps.shape, 1.0 / (gaps.size * bin_width))
        axis.hist(
            gaps,
            bins=edges,
            weights=weights,
            color="#5B8FF9",
            alpha=0.72,
            label="Эксперимент",
        )
        axis.plot(
            curve_x,
            wigner_density(curve_x),
            color="#D64545",
            linewidth=2.2,
            label="Догадка Вигнера",
        )
        axis.set_title(f'{result["size"]} x {result["size"]}')
        axis.set_xlabel("Нормированный интервал s")
        axis.set_xlim(0.0, HISTOGRAM_LIMIT)
        axis.grid(alpha=0.2)
        axis.text(
            0.97,
            0.95,
            f'D_KS = {result["ks_distance"]:.4f}\n'
            f'P(s < 0.1) = {result["near_zero_fraction"]:.4f}',
            transform=axis.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axes[0].set_ylabel("Плотность")
    axes[0].legend(loc="upper left")
    figure.suptitle(title, y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_summary(name, results):
    print(name)
    print("size  raw mean gap   D_KS     P(s < 0.1)  P(s = 0)")
    for result in results:
        print(
            f'{result["size"]:>4}  '
            f'{result["raw_mean"]:>12.6f}  '
            f'{result["ks_distance"]:>7.4f}  '
            f'{result["near_zero_fraction"]:>10.4f}  '
            f'{result["zero_fraction"]:>8.4f}'
        )
    print()


def print_exact_rademacher_result():
    mean_gap = 2.0 + np.sqrt(2.0)
    normalized_values = (
        0.0,
        4.0 / mean_gap,
        4.0 * np.sqrt(2.0) / mean_gap,
    )
    probabilities = (0.25, 0.5, 0.25)
    print("Exact 2 x 2 Rademacher distribution")
    for value, probability in zip(normalized_values, probabilities):
        print(f"s = {value:.10f}, probability = {probability:.2f}")


def main():
    output_directory = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)

    gaussian = analyze_ensemble(rng, "gaussian")
    rademacher = analyze_ensemble(rng, "rademacher")

    plot_results(
        gaussian,
        "Гауссовы симметричные матрицы",
        output_directory / "gaussian_spacing.png",
    )
    plot_results(
        rademacher,
        "Симметричные матрицы с исходными элементами -1 и 1",
        output_directory / "rademacher_spacing.png",
    )

    print_summary("Gaussian ensemble", gaussian)
    print_summary("Rademacher ensemble", rademacher)
    print_exact_rademacher_result()


if __name__ == "__main__":
    main()

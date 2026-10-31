"""Multiple linear regression for the lipophilicity data set."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2, f as f_distribution, jarque_bera, t as t_distribution


DATA_FILE = Path(__file__).with_name("lipo.csv")
COEFFICIENTS_FILE = Path(__file__).with_name("coefficients.csv")
RESIDUALS_FILE = Path(__file__).with_name("residuals.csv")
CONFIDENCE_LEVEL = 0.95


def load_data(path: Path) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Read names, response values and 24 regressors from a CSV file."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    if not rows:
        raise ValueError("The data file is empty")

    columns = list(rows[0])
    if len(columns) < 3 or columns[0] != "Вещество" or columns[1] != "logP":
        raise ValueError("Unexpected CSV structure")

    regressors = columns[2:]
    substances = [row["Вещество"] for row in rows]
    response = np.array([float(row["logP"]) for row in rows], dtype=float)
    features = np.array(
        [[float(row[name]) for name in regressors] for row in rows], dtype=float
    )

    if not np.isfinite(response).all() or not np.isfinite(features).all():
        raise ValueError("All numerical values must be finite")

    return substances, regressors, response, features


def variance_inflation_factors(features: np.ndarray) -> np.ndarray:
    """Calculate one VIF value for every feature column."""
    row_count, column_count = features.shape
    values = np.empty(column_count, dtype=float)

    for index in range(column_count):
        response = features[:, index]
        other_features = np.delete(features, index, axis=1)
        design = np.column_stack([np.ones(row_count), other_features])
        estimates, *_ = np.linalg.lstsq(design, response, rcond=None)
        residuals = response - design @ estimates
        total_sum = np.sum((response - response.mean()) ** 2)
        r_squared = 1.0 - float(residuals @ residuals) / total_sum
        values[index] = 1.0 / (1.0 - r_squared)

    return values


def write_coefficients(
    names: list[str],
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    t_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    standardized: np.ndarray,
) -> None:
    """Write coefficient estimates and confidence intervals."""
    with COEFFICIENTS_FILE.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(
            [
                "regressor",
                "estimate",
                "standard_error",
                "t_value",
                "ci_95_low",
                "ci_95_high",
                "standardized_coefficient",
            ]
        )
        for index, name in enumerate(names):
            standardized_value = "" if index == 0 else f"{standardized[index - 1]:.10f}"
            writer.writerow(
                [
                    name,
                    f"{estimates[index]:.10f}",
                    f"{standard_errors[index]:.10f}",
                    f"{t_values[index]:.10f}",
                    f"{lower_bounds[index]:.10f}",
                    f"{upper_bounds[index]:.10f}",
                    standardized_value,
                ]
            )


def write_residuals(
    substances: list[str],
    observed: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
) -> None:
    """Write all observed, fitted and residual values."""
    with RESIDUALS_FILE.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["substance", "observed_logP", "fitted_logP", "residual"])
        for name, actual, prediction, residual in zip(
            substances, observed, fitted, residuals
        ):
            writer.writerow(
                [name, f"{actual:.10f}", f"{prediction:.10f}", f"{residual:.10f}"]
            )


def main() -> None:
    substances, regressors, response, features = load_data(DATA_FILE)
    observation_count = len(response)
    design = np.column_stack([np.ones(observation_count), features])
    parameter_count = design.shape[1]
    predictor_count = parameter_count - 1
    residual_df = observation_count - parameter_count
    rank = int(np.linalg.matrix_rank(design))

    if rank != parameter_count:
        raise ValueError("The design matrix does not have full column rank")
    if residual_df <= 0:
        raise ValueError("There are not enough observations to estimate the variance")

    estimates, *_ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ estimates
    residuals = response - fitted
    residual_sum_squares = float(residuals @ residuals)
    total_sum_squares = float(np.sum((response - response.mean()) ** 2))
    error_variance = residual_sum_squares / residual_df
    regression_standard_error = math.sqrt(error_variance)
    root_mean_square_error = math.sqrt(residual_sum_squares / observation_count)
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (
        observation_count - 1
    ) / residual_df

    xtx_inverse = np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(error_variance * np.diag(xtx_inverse))
    t_values = estimates / standard_errors
    critical_t = float(
        t_distribution.ppf((1.0 + CONFIDENCE_LEVEL) / 2.0, residual_df)
    )
    lower_bounds = estimates - critical_t * standard_errors
    upper_bounds = estimates + critical_t * standard_errors

    feature_standard_deviations = features.std(axis=0, ddof=1)
    response_standard_deviation = response.std(ddof=1)
    standardized = (
        estimates[1:] * feature_standard_deviations / response_standard_deviation
    )

    f_statistic = (r_squared / predictor_count) / (
        (1.0 - r_squared) / residual_df
    )
    model_p_value = float(
        f_distribution.sf(f_statistic, predictor_count, residual_df)
    )

    squared_residuals = residuals**2
    auxiliary_estimates, *_ = np.linalg.lstsq(
        design, squared_residuals, rcond=None
    )
    auxiliary_residuals = squared_residuals - design @ auxiliary_estimates
    auxiliary_total = float(
        np.sum((squared_residuals - squared_residuals.mean()) ** 2)
    )
    auxiliary_r_squared = 1.0 - float(
        auxiliary_residuals @ auxiliary_residuals
    ) / auxiliary_total
    breusch_pagan = observation_count * auxiliary_r_squared
    breusch_pagan_p = float(chi2.sf(breusch_pagan, predictor_count))

    jarque_bera_result = jarque_bera(residuals)

    reset_design = np.column_stack([design, fitted**2, fitted**3])
    reset_estimates, *_ = np.linalg.lstsq(reset_design, response, rcond=None)
    reset_residuals = response - reset_design @ reset_estimates
    reset_rss = float(reset_residuals @ reset_residuals)
    added_terms = 2
    reset_df = observation_count - reset_design.shape[1]
    reset_f = ((residual_sum_squares - reset_rss) / added_terms) / (
        reset_rss / reset_df
    )
    reset_p = float(f_distribution.sf(reset_f, added_terms, reset_df))

    standardized_design = np.column_stack(
        [
            np.ones(observation_count),
            (features - features.mean(axis=0)) / feature_standard_deviations,
        ]
    )
    condition_number = float(np.linalg.cond(standardized_design))
    vifs = variance_inflation_factors(features)
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)

    coefficient_names = ["intercept", *regressors]
    write_coefficients(
        coefficient_names,
        estimates,
        standard_errors,
        t_values,
        lower_bounds,
        upper_bounds,
        standardized,
    )
    write_residuals(substances, response, fitted, residuals)

    print("Linear regression for logP")
    print(f"Observations: {observation_count}")
    print(f"Parameters including intercept: {parameter_count}")
    print(f"Residual degrees of freedom: {residual_df}")
    print(f"Design matrix rank: {rank}")
    print()
    print("Fit statistics")
    print(f"RSS: {residual_sum_squares:.10f}")
    print(f"Error variance s^2: {error_variance:.10f}")
    print(f"Regression standard error s: {regression_standard_error:.10f}")
    print(f"RMSE: {root_mean_square_error:.10f}")
    print(f"R^2: {r_squared:.10f}")
    print(f"Adjusted R^2: {adjusted_r_squared:.10f}")
    print(f"F({predictor_count}, {residual_df}): {f_statistic:.6f}")
    print(f"Model p-value: {model_p_value:.6g}")
    print()
    print(f"Coefficient estimates and {CONFIDENCE_LEVEL:.0%} confidence intervals")
    print(f"Critical t value: {critical_t:.10f}")
    print(f"{'name':<10} {'estimate':>12} {'std. error':>12} {'CI low':>12} {'CI high':>12}")
    for name, estimate, standard_error, lower, upper in zip(
        coefficient_names,
        estimates,
        standard_errors,
        lower_bounds,
        upper_bounds,
    ):
        print(
            f"{name:<10} {estimate:12.6f} {standard_error:12.6f} "
            f"{lower:12.6f} {upper:12.6f}"
        )

    print()
    print("Largest absolute standardized coefficients")
    for index in np.argsort(np.abs(standardized))[::-1][:5]:
        print(f"{regressors[index]:<10} {standardized[index]: .6f}")

    print()
    print("Diagnostics")
    print(f"Standardized condition number: {condition_number:.6f}")
    print(
        f"Breusch-Pagan LM({predictor_count}): {breusch_pagan:.6f}, "
        f"p = {breusch_pagan_p:.6f}"
    )
    print(
        f"Jarque-Bera: {jarque_bera_result.statistic:.6f}, "
        f"p = {jarque_bera_result.pvalue:.6f}"
    )
    print(f"RESET F(2, {reset_df}): {reset_f:.6f}, p = {reset_p:.6f}")
    print("Largest VIF values:")
    for index in np.argsort(vifs)[::-1][:5]:
        print(f"{regressors[index]:<10} {vifs[index]:.3f}")
    print("Observations with leverage at least 0.9:")
    for index in np.flatnonzero(leverage >= 0.9):
        print(f"{substances[index]:<20} {leverage[index]:.6f}")

    print()
    print(f"Coefficient table: {COEFFICIENTS_FILE.name}")
    print(f"Residual table: {RESIDUALS_FILE.name}")


if __name__ == "__main__":
    main()

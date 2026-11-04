"""Two-layer perceptron for the lipophilicity data set."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor


DATA_FILE = Path(__file__).with_name("lipo.csv")
WEIGHTS_FILE = Path(__file__).with_name("weights.json")
HIDDEN_LAYER_SIZES = (2, 4, 8, 16, 32)
LEARNING_RATE = 0.03
TOLERANCE = 0.005
MAX_ITERATIONS = 50_000
DECAY_SCALE = 50_000.0
RANDOM_SEED = 42
INITIALIZATION_SEEDS = (41, 42, 43)
CROSS_VALIDATION_FOLDS = 5


@dataclass
class Parameters:
    """Weights and biases of a network with one hidden layer."""

    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float


@dataclass
class TrainingResult:
    """Parameters and diagnostics returned by gradient descent."""

    parameters: Parameters
    iterations: int
    converged: bool
    gradient_norm: float
    rmse: float
    r_squared: float


@dataclass
class ValidationResult:
    """Out-of-fold regression metrics."""

    rmse: float
    r_squared: float


@dataclass
class ValidationSummary:
    """Mean and standard deviation across initial parameter values."""

    mean_rmse: float
    std_rmse: float
    mean_r_squared: float
    std_r_squared: float
    rmse_values: tuple[float, ...]
    r_squared_values: tuple[float, ...]


@dataclass
class FoldData:
    """Prepared training and test data for one validation fold."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    standardized_train: np.ndarray
    standardized_test: np.ndarray
    standardized_response: np.ndarray
    response_mean: float
    response_scale: float


def load_data(path: Path) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Read substance names, logP values and molecular descriptors."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    if not rows:
        raise ValueError("The data file is empty")

    columns = list(rows[0])
    if len(columns) != 26 or columns[0] != "Вещество" or columns[1] != "logP":
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


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns and return the transformed values and their scale."""
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0)
    if np.any(standard_deviations == 0.0):
        raise ValueError("A variable with zero variance cannot be standardized")
    return (values - means) / standard_deviations, means, standard_deviations


def initialize_parameters(
    feature_count: int,
    hidden_units: int,
    random_seed: int = RANDOM_SEED,
) -> Parameters:
    """Create deterministic He-style initial values."""
    generator = np.random.default_rng(random_seed)
    input_weights = generator.normal(
        0.0,
        math.sqrt(2.0 / feature_count),
        size=(hidden_units, feature_count),
    )
    output_weights = generator.normal(
        0.0, math.sqrt(2.0 / hidden_units), size=hidden_units
    )
    return Parameters(
        input_weights=input_weights,
        hidden_bias=np.zeros(hidden_units, dtype=float),
        output_weights=output_weights,
        output_bias=0.0,
    )


def predict(features: np.ndarray, parameters: Parameters) -> np.ndarray:
    """Calculate the standardized response of the perceptron."""
    hidden_input = features @ parameters.input_weights.T + parameters.hidden_bias
    hidden_output = np.maximum(hidden_input, 0.0)
    return hidden_output @ parameters.output_weights + parameters.output_bias


def descent_direction(
    features: np.ndarray,
    response: np.ndarray,
    parameters: Parameters,
) -> tuple[Parameters, float]:
    """Calculate the negative gradient of half the mean squared error."""
    hidden_input = features @ parameters.input_weights.T + parameters.hidden_bias
    hidden_output = np.maximum(hidden_input, 0.0)
    residuals = response - (
        hidden_output @ parameters.output_weights + parameters.output_bias
    )
    observation_count = response.size

    output_bias = float(residuals.mean())
    output_weights = hidden_output.T @ residuals / observation_count
    hidden_terms = (
        residuals[:, None]
        * parameters.output_weights[None, :]
        * (hidden_input >= 0.0)
    )
    hidden_bias = hidden_terms.mean(axis=0)
    input_weights = hidden_terms.T @ features / observation_count

    norm = math.sqrt(
        output_bias**2
        + float(output_weights @ output_weights)
        + float(hidden_bias @ hidden_bias)
        + float(np.sum(input_weights**2))
    )
    direction = Parameters(input_weights, hidden_bias, output_weights, output_bias)
    return direction, norm


def update_parameters(
    parameters: Parameters, direction: Parameters, learning_rate: float
) -> None:
    """Perform one full-batch gradient descent update."""
    parameters.input_weights += learning_rate * direction.input_weights
    parameters.hidden_bias += learning_rate * direction.hidden_bias
    parameters.output_weights += learning_rate * direction.output_weights
    parameters.output_bias += learning_rate * direction.output_bias


def regression_metrics(actual: np.ndarray, fitted: np.ndarray) -> tuple[float, float]:
    """Return RMSE and the coefficient of determination."""
    residual_sum = float(np.sum((actual - fitted) ** 2))
    total_sum = float(np.sum((actual - actual.mean()) ** 2))
    return math.sqrt(residual_sum / actual.size), 1.0 - residual_sum / total_sum


def train_perceptron(
    features: np.ndarray,
    response: np.ndarray,
    response_mean: float,
    response_scale: float,
    hidden_units: int,
    decreasing_rate: bool,
    random_seed: int = RANDOM_SEED,
) -> TrainingResult:
    """Fit a perceptron by full-batch gradient descent."""
    parameters = initialize_parameters(
        features.shape[1], hidden_units, random_seed=random_seed
    )
    converged = False
    gradient_norm = math.inf

    for iteration in range(MAX_ITERATIONS + 1):
        direction, gradient_norm = descent_direction(features, response, parameters)
        if gradient_norm < TOLERANCE:
            converged = True
            break
        if iteration == MAX_ITERATIONS:
            break

        learning_rate = LEARNING_RATE
        if decreasing_rate:
            learning_rate /= 1.0 + iteration / DECAY_SCALE
        update_parameters(parameters, direction, learning_rate)

    fitted = response_mean + response_scale * predict(features, parameters)
    actual = response_mean + response_scale * response
    rmse, r_squared = regression_metrics(actual, fitted)
    return TrainingResult(
        parameters=parameters,
        iterations=iteration,
        converged=converged,
        gradient_norm=gradient_norm,
        rmse=rmse,
        r_squared=r_squared,
    )


def local_sensitivity_values(
    features: np.ndarray,
    parameters: Parameters,
    response_scale: float,
) -> np.ndarray:
    """Calculate signed local derivatives for every observation and feature."""
    hidden_input = features @ parameters.input_weights.T + parameters.hidden_bias
    local_gradients = (
        (hidden_input >= 0.0) * parameters.output_weights[None, :]
    ) @ parameters.input_weights
    return response_scale * local_gradients


def prepare_validation_folds(
    features: np.ndarray, response: np.ndarray
) -> tuple[list[FoldData], int, int]:
    """Prepare fold-specific scaling without using test observations."""
    splitter = KFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    folds: list[FoldData] = []
    folds_with_constant_regressors = 0
    maximum_constant_regressors = 0

    for train_indices, test_indices in splitter.split(features):
        train_features = features[train_indices]
        feature_means = train_features.mean(axis=0)
        feature_scales = train_features.std(axis=0)
        constant_count = int(np.sum(feature_scales == 0.0))
        if constant_count:
            folds_with_constant_regressors += 1
            maximum_constant_regressors = max(
                maximum_constant_regressors, constant_count
            )
            feature_scales[feature_scales == 0.0] = 1.0

        standardized_train = (train_features - feature_means) / feature_scales
        standardized_test = (features[test_indices] - feature_means) / feature_scales
        response_mean = float(response[train_indices].mean())
        response_scale = float(response[train_indices].std())
        standardized_response = (
            response[train_indices] - response_mean
        ) / response_scale
        folds.append(
            FoldData(
                train_indices=train_indices,
                test_indices=test_indices,
                standardized_train=standardized_train,
                standardized_test=standardized_test,
                standardized_response=standardized_response,
                response_mean=response_mean,
                response_scale=response_scale,
            )
        )

    return folds, folds_with_constant_regressors, maximum_constant_regressors


def summarize_validation(results: list[ValidationResult]) -> ValidationSummary:
    """Summarize validation metrics obtained from several initial values."""
    rmse_values = np.array([result.rmse for result in results], dtype=float)
    r_squared_values = np.array(
        [result.r_squared for result in results], dtype=float
    )
    return ValidationSummary(
        mean_rmse=float(rmse_values.mean()),
        std_rmse=float(rmse_values.std()),
        mean_r_squared=float(r_squared_values.mean()),
        std_r_squared=float(r_squared_values.std()),
        rmse_values=tuple(float(value) for value in rmse_values),
        r_squared_values=tuple(float(value) for value in r_squared_values),
    )


def cross_validate_manual_and_linear(
    features: np.ndarray,
    response: np.ndarray,
) -> tuple[dict[int, ValidationSummary], ValidationResult, int, int]:
    """Validate manual models over several initial parameter values."""
    (
        folds,
        folds_with_constant_regressors,
        maximum_constant_regressors,
    ) = prepare_validation_folds(features, response)
    results_by_size: dict[int, list[ValidationResult]] = {
        hidden_units: [] for hidden_units in HIDDEN_LAYER_SIZES
    }

    for random_seed in INITIALIZATION_SEEDS:
        manual_fitted = {
            hidden_units: np.empty_like(response)
            for hidden_units in HIDDEN_LAYER_SIZES
        }
        for fold in folds:
            for hidden_units in HIDDEN_LAYER_SIZES:
                result = train_perceptron(
                    fold.standardized_train,
                    fold.standardized_response,
                    fold.response_mean,
                    fold.response_scale,
                    hidden_units,
                    decreasing_rate=False,
                    random_seed=random_seed,
                )
                manual_fitted[hidden_units][fold.test_indices] = (
                    fold.response_mean
                    + fold.response_scale
                    * predict(fold.standardized_test, result.parameters)
                )

        for hidden_units, fitted in manual_fitted.items():
            results_by_size[hidden_units].append(
                ValidationResult(*regression_metrics(response, fitted))
            )

    linear_fitted = np.empty_like(response)
    for fold in folds:
        linear_design = np.column_stack(
            [np.ones(fold.train_indices.size), features[fold.train_indices]]
        )
        coefficients, *_ = np.linalg.lstsq(
            linear_design, response[fold.train_indices], rcond=None
        )
        linear_test_design = np.column_stack(
            [np.ones(fold.test_indices.size), features[fold.test_indices]]
        )
        linear_fitted[fold.test_indices] = linear_test_design @ coefficients

    manual_results = {
        hidden_units: summarize_validation(results)
        for hidden_units, results in results_by_size.items()
    }
    linear_result = ValidationResult(*regression_metrics(response, linear_fitted))
    return (
        manual_results,
        linear_result,
        folds_with_constant_regressors,
        maximum_constant_regressors,
    )


def cross_validate_sklearn(
    features: np.ndarray,
    response: np.ndarray,
) -> dict[int, ValidationSummary]:
    """Validate all library models over several initial parameter values."""
    folds, _, _ = prepare_validation_folds(features, response)
    results_by_size: dict[int, list[ValidationResult]] = {
        hidden_units: [] for hidden_units in HIDDEN_LAYER_SIZES
    }

    for random_seed in INITIALIZATION_SEEDS:
        fitted_by_size = {
            hidden_units: np.empty_like(response)
            for hidden_units in HIDDEN_LAYER_SIZES
        }
        for fold in folds:
            for hidden_units in HIDDEN_LAYER_SIZES:
                model = MLPRegressor(
                    hidden_layer_sizes=(hidden_units,),
                    activation="relu",
                    solver="lbfgs",
                    alpha=0.0,
                    max_iter=MAX_ITERATIONS,
                    tol=1e-6,
                    random_state=random_seed,
                )
                model.fit(fold.standardized_train, fold.standardized_response)
                fitted_by_size[hidden_units][fold.test_indices] = (
                    fold.response_mean
                    + fold.response_scale * model.predict(fold.standardized_test)
                )

        for hidden_units, fitted in fitted_by_size.items():
            results_by_size[hidden_units].append(
                ValidationResult(*regression_metrics(response, fitted))
            )

    return {
        hidden_units: summarize_validation(results)
        for hidden_units, results in results_by_size.items()
    }


def fit_linear_regression(
    features: np.ndarray, response: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Fit the linear reference model and standardize its coefficients."""
    design = np.column_stack([np.ones(response.size), features])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    rmse, r_squared = regression_metrics(response, design @ coefficients)
    standardized_coefficients = (
        coefficients[1:] * features.std(axis=0) / response.std()
    )
    return rmse, r_squared, standardized_coefficients


def fit_sklearn_model(
    features: np.ndarray,
    response: np.ndarray,
    response_mean: float,
    response_scale: float,
    hidden_units: int,
) -> tuple[int, float, float]:
    """Fit an equivalent scikit-learn model with the L-BFGS solver."""
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="lbfgs",
        alpha=0.0,
        max_iter=MAX_ITERATIONS,
        tol=1e-6,
        random_state=RANDOM_SEED,
    )
    model.fit(features, response)
    fitted = response_mean + response_scale * model.predict(features)
    actual = response_mean + response_scale * response
    rmse, r_squared = regression_metrics(actual, fitted)
    return model.n_iter_, rmse, r_squared


def write_weights(
    regressors: list[str],
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    response_mean: float,
    response_scale: float,
    fixed_results: dict[int, TrainingResult],
    decreasing_results: dict[int, TrainingResult],
) -> None:
    """Write all fitted parameters in a readable format."""
    document: dict[str, object] = {
        "regressors": regressors,
        "model": "b2 + w2^T relu(W1 x_standardized + b1)",
        "input_scale": "x_standardized = (x - feature_mean) / feature_scale",
        "output_scale": "logP = response_mean + response_scale * y_standardized",
        "learning_rate": LEARNING_RATE,
        "tolerance": TOLERANCE,
        "max_iterations": MAX_ITERATIONS,
        "initialization_seed": RANDOM_SEED,
        "decreasing_learning_rate": (
            "learning_rate / (1 + iteration / decay_scale)"
        ),
        "decay_scale": DECAY_SCALE,
        "standardization": {
            "feature_mean": feature_means.tolist(),
            "feature_scale": feature_scales.tolist(),
            "response_mean": response_mean,
            "response_scale": response_scale,
        },
        "models": {},
    }
    models = document["models"]
    assert isinstance(models, dict)

    for schedule, results in (
        ("fixed", fixed_results),
        ("decreasing", decreasing_results),
    ):
        schedule_data: dict[str, object] = {}
        for hidden_units, result in results.items():
            parameters = result.parameters
            schedule_data[str(hidden_units)] = {
                "input_weights": parameters.input_weights.tolist(),
                "hidden_bias": parameters.hidden_bias.tolist(),
                "output_weights": parameters.output_weights.tolist(),
                "output_bias": parameters.output_bias,
            }
        models[schedule] = schedule_data

    with WEIGHTS_FILE.open("w", encoding="utf-8") as target:
        json.dump(document, target, ensure_ascii=False, indent=2)
        target.write("\n")


def print_results(title: str, results: dict[int, TrainingResult]) -> None:
    """Print one comparison table."""
    print(title)
    print(
        f"{'hidden':>6} {'parameters':>10} {'iterations':>10} "
        f"{'converged':>10} {'|gradient|':>12} {'RMSE':>10} {'R^2':>10}"
    )
    feature_count = 24
    for hidden_units, result in results.items():
        parameter_count = hidden_units * (feature_count + 2) + 1
        print(
            f"{hidden_units:6d} {parameter_count:10d} {result.iterations:10d} "
            f"{str(result.converged):>10} {result.gradient_norm:12.6f} "
            f"{result.rmse:10.6f} {result.r_squared:10.6f}"
        )
    print()


def main() -> None:
    substances, regressors, response, features = load_data(DATA_FILE)
    standardized_features, feature_means, feature_scales = standardize(features)
    standardized_response, response_mean_array, response_scale_array = standardize(
        response[:, None]
    )
    standardized_response = standardized_response[:, 0]
    response_mean = float(response_mean_array[0])
    response_scale = float(response_scale_array[0])

    fixed_results = {
        hidden_units: train_perceptron(
            standardized_features,
            standardized_response,
            response_mean,
            response_scale,
            hidden_units,
            decreasing_rate=False,
        )
        for hidden_units in HIDDEN_LAYER_SIZES
    }
    decreasing_results = {
        hidden_units: train_perceptron(
            standardized_features,
            standardized_response,
            response_mean,
            response_scale,
            hidden_units,
            decreasing_rate=True,
        )
        for hidden_units in HIDDEN_LAYER_SIZES
    }

    write_weights(
        regressors,
        feature_means,
        feature_scales,
        response_mean,
        response_scale,
        fixed_results,
        decreasing_results,
    )
    (
        manual_validation,
        linear_validation,
        folds_with_constant_regressors,
        maximum_constant_regressors,
    ) = cross_validate_manual_and_linear(features, response)
    selected_hidden_units = min(
        manual_validation, key=lambda size: manual_validation[size].mean_rmse
    )
    sklearn_validation = cross_validate_sklearn(features, response)
    selected_result = fixed_results[selected_hidden_units]
    signed_sensitivity_values = local_sensitivity_values(
        standardized_features,
        selected_result.parameters,
        response_scale,
    )
    sensitivity_values = np.abs(signed_sensitivity_values)
    sensitivity_means = sensitivity_values.mean(axis=0)
    signed_sensitivity_means = signed_sensitivity_values.mean(axis=0)
    sensitivity_medians = np.median(sensitivity_values, axis=0)
    sensitivity_standard_deviations = sensitivity_values.std(axis=0)
    sensitivity_maxima = sensitivity_values.max(axis=0)
    sensitivity_order = np.argsort(sensitivity_means)[::-1]

    linear_rmse, linear_r_squared, linear_standardized = fit_linear_regression(
        features, response
    )
    sklearn_training_results = {
        hidden_units: fit_sklearn_model(
            standardized_features,
            standardized_response,
            response_mean,
            response_scale,
            hidden_units,
        )
        for hidden_units in HIDDEN_LAYER_SIZES
    }

    print("Two-layer perceptron for logP")
    print(f"Observations: {len(substances)}")
    print(f"Regressors: {len(regressors)}")
    print(f"Learning rate epsilon: {LEARNING_RATE}")
    print(f"Tolerance delta: {TOLERANCE}")
    print(f"Iteration limit: {MAX_ITERATIONS}")
    print()
    print_results("Fixed learning rate", fixed_results)
    print_results("Decreasing learning rate", decreasing_results)

    print("Five-fold out-of-fold validation")
    print(f"Initial parameter seeds: {INITIALIZATION_SEEDS}")
    print(
        f"{'model':<22} {'RMSE mean':>10} {'RMSE sd':>10} "
        f"{'R^2 mean':>10} {'R^2 sd':>10}"
    )
    for hidden_units, result in manual_validation.items():
        print(
            f"manual, {hidden_units:2d} hidden   "
            f"{result.mean_rmse:10.6f} {result.std_rmse:10.6f} "
            f"{result.mean_r_squared:10.6f} {result.std_r_squared:10.6f}"
        )
    print(
        f"linear regression     {linear_validation.rmse:10.6f} {'-':>10} "
        f"{linear_validation.r_squared:10.6f} {'-':>10}"
    )
    for hidden_units, result in sklearn_validation.items():
        print(
            f"scikit-learn, {hidden_units:2d} hidden "
            f"{result.mean_rmse:10.6f} {result.std_rmse:10.6f} "
            f"{result.mean_r_squared:10.6f} {result.std_r_squared:10.6f}"
        )
    print("Seed-specific out-of-fold RMSE")
    for model_name, validation_results in (
        ("manual", manual_validation),
        ("scikit-learn", sklearn_validation),
    ):
        for hidden_units, result in validation_results.items():
            seed_values = ", ".join(
                f"{seed}={value:.6f}"
                for seed, value in zip(INITIALIZATION_SEEDS, result.rmse_values)
            )
            print(f"{model_name}, {hidden_units:2d} hidden: {seed_values}")
    print(
        f"Folds with constant training regressors: "
        f"{folds_with_constant_regressors} of {CROSS_VALIDATION_FOLDS}; "
        f"maximum constant regressors in one fold: {maximum_constant_regressors}"
    )
    print(
        f"Lowest mean manual out-of-fold RMSE in this split: "
        f"{selected_hidden_units} hidden units"
    )
    print()

    print(
        f"Local sensitivity for the {selected_hidden_units}-unit manual model"
    )
    print("Largest average absolute derivatives")
    print(
        f"{'feature':<10} {'mean abs':>10} {'mean signed':>12} "
        f"{'median abs':>12} {'sd abs':>10} {'maximum abs':>12}"
    )
    for index in sensitivity_order[:5]:
        print(
            f"{regressors[index]:<10} {sensitivity_means[index]:10.6f} "
            f"{signed_sensitivity_means[index]:12.6f} "
            f"{sensitivity_medians[index]:12.6f} "
            f"{sensitivity_standard_deviations[index]:10.6f} "
            f"{sensitivity_maxima[index]:12.6f}"
        )
    print("Smallest average absolute derivatives")
    print(f"{'feature':<10} {'mean abs':>10} {'mean signed':>12}")
    for index in sensitivity_order[-5:][::-1]:
        print(
            f"{regressors[index]:<10} {sensitivity_means[index]:10.6f} "
            f"{signed_sensitivity_means[index]:12.6f}"
        )
    print()

    print("Training-sample reference results")
    print(f"Linear regression: RMSE = {linear_rmse:.6f}, R^2 = {linear_r_squared:.6f}")
    for hidden_units, result in sklearn_training_results.items():
        iterations, rmse, r_squared = result
        print(
            f"scikit-learn MLP ({hidden_units} hidden units): "
            f"iterations = {iterations}, RMSE = {rmse:.6f}, "
            f"R^2 = {r_squared:.6f}"
        )
    print("Largest absolute standardized linear coefficients")
    for index in np.argsort(np.abs(linear_standardized))[::-1][:5]:
        print(f"{regressors[index]:<10} {linear_standardized[index]: .6f}")
    correlation_matrix = np.corrcoef(features, rowvar=False)
    hba2_index = regressors.index("HBA2")
    print("Correlations with HBA2")
    for name in ("PSA", "HBA1", "MR", "C", "RINGS"):
        index = regressors.index(name)
        print(f"{name:<10} {correlation_matrix[hba2_index, index]: .6f}")
    print()
    print(f"Fitted weights: {WEIGHTS_FILE.name}")


if __name__ == "__main__":
    main()

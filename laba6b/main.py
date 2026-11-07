"""Numerical optimization and gradient methods for three regression models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIRECTORY = Path(__file__).parent
X = np.array(
    [0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90],
    dtype=float,
)
Y = np.array(
    [0.67, 0.85, 1.05, 1.00, 1.40, 1.50, 1.30, 1.54, 1.55, 1.68, 1.73, 1.60],
    dtype=float,
)

RANDOM_SEED = 42
DIFFERENCE_STEP = 1e-5
GRADIENT_TOLERANCE = 1e-7
BATCH_SIZES = (1, 3, 6, 12)
STOCHASTIC_EPOCHS = 4_000
COMPARISON_BATCH_SIZE = 3


@dataclass(frozen=True)
class MinimumResult:
    """Result returned by one-dimensional minimization."""

    argument: float
    value: float
    iterations: int


@dataclass(frozen=True)
class ModelSpec:
    """Definition and optimization settings of a regression model."""

    name: str
    parameter_names: tuple[str, ...]
    initial_parameters: tuple[float, ...]
    learning_rate: float
    adam_learning_rate: float
    maximum_iterations: int
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class FitResult:
    """Parameters and diagnostics produced by an optimizer."""

    parameters: np.ndarray
    loss: float
    iterations: int
    converged: bool
    loss_history: np.ndarray


def objective_function(z: np.ndarray | float) -> np.ndarray | float:
    """Evaluate the one-dimensional function."""
    return (
        1.0
        - 0.5 * np.exp(-10.0 * (np.asarray(z) - 0.65) ** 2)
        - 0.45 * np.exp(-50.0 * (np.asarray(z) - 0.35) ** 2)
    )


def golden_section_minimize(
    function: Callable[[float], float],
    left: float,
    right: float,
    tolerance: float = 1e-12,
    maximum_iterations: int = 1_000,
) -> MinimumResult:
    """Find a minimum of a unimodal function on a closed interval."""
    if not left < right:
        raise ValueError("left must be smaller than right")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    first_value = function(first)
    second_value = function(second)

    for iteration in range(1, maximum_iterations + 1):
        if right - left <= tolerance:
            break
        if first_value <= second_value:
            right = second
            second = first
            second_value = first_value
            first = right - ratio * (right - left)
            first_value = function(first)
        else:
            left = first
            first = second
            first_value = second_value
            second = left + ratio * (right - left)
            second_value = function(second)
    else:
        raise RuntimeError("golden-section search did not converge")

    argument = (left + right) / 2.0
    return MinimumResult(argument, function(argument), iteration)


def linear_prediction(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate w1*x + b1."""
    weight, bias = parameters
    return weight * x + bias


def quadratic_prediction(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate w1*x^2 + w2*x + b1."""
    quadratic_weight, linear_weight, bias = parameters
    return quadratic_weight * x**2 + linear_weight * x + bias


def perceptron_prediction(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a two-hidden-unit ReLU perceptron."""
    w11, b11, w12, b12, w21, w22, b21 = parameters
    first_hidden = np.maximum(w11 * x + b11, 0.0)
    second_hidden = np.maximum(w12 * x + b12, 0.0)
    return np.maximum(w21 * first_hidden + w22 * second_hidden + b21, 0.0)


MODELS = (
    ModelSpec(
        name="linear",
        parameter_names=("w1", "b1"),
        initial_parameters=(0.5, 0.5),
        learning_rate=0.025,
        adam_learning_rate=0.015,
        maximum_iterations=20_000,
        predict=linear_prediction,
    ),
    ModelSpec(
        name="quadratic",
        parameter_names=("w1", "w2", "b1"),
        initial_parameters=(0.0, 0.5, 0.5),
        learning_rate=0.006,
        adam_learning_rate=0.012,
        maximum_iterations=40_000,
        predict=quadratic_prediction,
    ),
    ModelSpec(
        name="perceptron",
        parameter_names=("w11", "b11", "w12", "b12", "w21", "w22", "b21"),
        initial_parameters=(0.8, 0.1, -0.6, 0.9, 0.7, -0.2, 0.6),
        learning_rate=0.002,
        adam_learning_rate=0.006,
        maximum_iterations=50_000,
        predict=perceptron_prediction,
    ),
)


def squared_error(
    model: ModelSpec,
    parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Return the sum of squared residuals."""
    residuals = y - model.predict(parameters, x)
    return float(residuals @ residuals)


def numerical_gradient(
    function: Callable[[np.ndarray], float], parameters: np.ndarray
) -> np.ndarray:
    """Calculate a central finite-difference gradient."""
    gradient = np.empty_like(parameters)
    for index, value in enumerate(parameters):
        step = DIFFERENCE_STEP * max(1.0, abs(float(value)))
        parameters[index] = value + step
        right_value = function(parameters)
        parameters[index] = value - step
        left_value = function(parameters)
        parameters[index] = value
        gradient[index] = (right_value - left_value) / (2.0 * step)
    return gradient


def fit_gradient_descent(model: ModelSpec) -> FitResult:
    """Fit a model with full-batch gradient descent."""
    parameters = np.array(model.initial_parameters, dtype=float)
    history = [squared_error(model, parameters, X, Y)]
    converged = False

    for iteration in range(1, model.maximum_iterations + 1):
        gradient = numerical_gradient(
            lambda values: squared_error(model, values, X, Y), parameters
        )
        parameters -= model.learning_rate * gradient
        history.append(squared_error(model, parameters, X, Y))
        if np.linalg.norm(gradient) < GRADIENT_TOLERANCE:
            converged = True
            break
        if not np.isfinite(history[-1]):
            raise FloatingPointError(f"gradient descent diverged for {model.name}")

    return FitResult(
        parameters=parameters,
        loss=history[-1],
        iterations=iteration,
        converged=converged,
        loss_history=np.array(history),
    )


def iterate_batches(
    observation_count: int, batch_size: int, generator: np.random.Generator
) -> tuple[np.ndarray, ...]:
    """Shuffle observations and split them into one epoch of mini-batches."""
    order = generator.permutation(observation_count)
    return tuple(
        order[start : start + batch_size]
        for start in range(0, observation_count, batch_size)
    )


def fit_minibatch(
    model: ModelSpec,
    batch_size: int,
    optimizer: str,
    epochs: int = STOCHASTIC_EPOCHS,
) -> FitResult:
    """Fit a model with SGD, Nesterov momentum or Adam."""
    if optimizer not in {"sgd", "nesterov", "adam"}:
        raise ValueError("unknown optimizer")
    if batch_size <= 0 or batch_size > X.size:
        raise ValueError("invalid batch size")

    parameters = np.array(model.initial_parameters, dtype=float)
    generator = np.random.default_rng(RANDOM_SEED)
    history = [squared_error(model, parameters, X, Y)]
    velocity = np.zeros_like(parameters)
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    update = 0

    for epoch in range(1, epochs + 1):
        for indices in iterate_batches(X.size, batch_size, generator):
            batch_x = X[indices]
            batch_y = Y[indices]

            if optimizer == "nesterov":
                lookahead = parameters + 0.9 * velocity
                gradient = numerical_gradient(
                    lambda values: squared_error(
                        model, values, batch_x, batch_y
                    ),
                    lookahead,
                )
                velocity = 0.9 * velocity - model.learning_rate * gradient
                parameters += velocity
            else:
                gradient = numerical_gradient(
                    lambda values: squared_error(
                        model, values, batch_x, batch_y
                    ),
                    parameters,
                )
                if optimizer == "sgd":
                    parameters -= model.learning_rate * gradient
                else:
                    update += 1
                    first_moment = 0.9 * first_moment + 0.1 * gradient
                    second_moment = 0.999 * second_moment + 0.001 * gradient**2
                    corrected_first = first_moment / (1.0 - 0.9**update)
                    corrected_second = second_moment / (1.0 - 0.999**update)
                    parameters -= (
                        model.adam_learning_rate
                        * corrected_first
                        / (np.sqrt(corrected_second) + 1e-8)
                    )

        loss = squared_error(model, parameters, X, Y)
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"{optimizer} diverged for {model.name}, batch {batch_size}"
            )
        history.append(loss)

    final_gradient = numerical_gradient(
        lambda values: squared_error(model, values, X, Y), parameters
    )
    return FitResult(
        parameters=parameters,
        loss=history[-1],
        iterations=epoch,
        converged=bool(np.linalg.norm(final_gradient) < GRADIENT_TOLERANCE),
        loss_history=np.array(history),
    )


def save_function_plot(minimum: MinimumResult) -> None:
    """Plot the one-dimensional function and its minimum."""
    grid = np.linspace(0.0, 1.0, 800)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(grid, objective_function(grid), color="#2457a6", label="f(z)")
    axis.scatter(
        [minimum.argument],
        [minimum.value],
        color="#c33c3c",
        zorder=3,
        label="minimum",
    )
    axis.set_xlabel("z")
    axis.set_ylabel("f(z)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "function_minimum.png", dpi=180)
    plt.close(figure)


def save_model_plot(results: dict[str, FitResult]) -> None:
    """Plot the three full-batch fitted dependencies."""
    grid = np.linspace(0.0, 2.0, 500)
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    for axis, model in zip(axes, MODELS):
        result = results[model.name]
        axis.scatter(X, Y, color="#222222", label="observations")
        axis.plot(
            grid,
            model.predict(result.parameters, grid),
            color="#2457a6",
            label="fit",
        )
        axis.set_title(model.name)
        axis.set_xlabel("x")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("y")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "model_fits.png", dpi=180)
    plt.close(figure)


def save_batch_plot(results: dict[tuple[str, int], FitResult]) -> None:
    """Plot SGD loss histories for different batch sizes."""
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for axis, model in zip(axes, MODELS):
        for batch_size in BATCH_SIZES:
            history = results[(model.name, batch_size)].loss_history
            axis.plot(history, label=f"batch {batch_size}", linewidth=1.2)
        axis.set_title(model.name)
        axis.set_xlabel("epoch")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("full SSE")
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "batch_size_comparison.png", dpi=180)
    plt.close(figure)


def save_optimizer_plot(
    full_results: dict[str, FitResult],
    stochastic_results: dict[str, dict[str, FitResult]],
) -> None:
    """Plot convergence histories for the compared optimizers."""
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    batches_per_epoch = math.ceil(X.size / COMPARISON_BATCH_SIZE)
    for axis, model in zip(axes, MODELS):
        full_history = full_results[model.name].loss_history
        axis.plot(
            np.arange(full_history.size),
            full_history,
            label="full GD",
            linewidth=1.2,
        )
        for optimizer in ("sgd", "nesterov", "adam"):
            history = stochastic_results[optimizer][model.name].loss_history
            axis.plot(
                np.arange(history.size) * batches_per_epoch,
                history,
                label=optimizer,
                linewidth=1.2,
            )
        axis.set_title(model.name)
        axis.set_xlabel("numerical gradient evaluations")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("full SSE")
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "optimizer_comparison.png", dpi=180)
    plt.close(figure)


def format_parameters(model: ModelSpec, parameters: np.ndarray) -> str:
    """Format fitted parameters as name-value pairs."""
    return ", ".join(
        f"{name}={value:.6f}"
        for name, value in zip(model.parameter_names, parameters)
    )


def main() -> None:
    minimum = golden_section_minimize(
        lambda value: float(objective_function(value)), 0.0, 1.0
    )
    full_results = {model.name: fit_gradient_descent(model) for model in MODELS}
    batch_results = {
        (model.name, batch_size): fit_minibatch(model, batch_size, "sgd")
        for model in MODELS
        for batch_size in BATCH_SIZES
    }
    stochastic_results = {
        optimizer: {
            model.name: fit_minibatch(
                model, COMPARISON_BATCH_SIZE, optimizer
            )
            for model in MODELS
        }
        for optimizer in ("sgd", "nesterov", "adam")
    }

    save_function_plot(minimum)
    save_model_plot(full_results)
    save_batch_plot(batch_results)
    save_optimizer_plot(full_results, stochastic_results)

    print("Golden-section minimum")
    print(f"z = {minimum.argument:.12f}")
    print(f"f(z) = {minimum.value:.12f}")
    print(f"iterations = {minimum.iterations}")
    print()

    print("Full-batch gradient descent")
    for model in MODELS:
        result = full_results[model.name]
        print(
            f"{model.name:<10} SSE={result.loss:.10f} "
            f"iterations={result.iterations} converged={result.converged}"
        )
        print(f"  {format_parameters(model, result.parameters)}")
    print()

    print("Mini-batch stochastic gradient descent")
    print(f"epochs = {STOCHASTIC_EPOCHS}")
    for model in MODELS:
        for batch_size in BATCH_SIZES:
            result = batch_results[(model.name, batch_size)]
            print(
                f"{model.name:<10} batch={batch_size:2d} "
                f"SSE={result.loss:.10f}"
            )
    print()

    print(f"Optimizer comparison, batch size = {COMPARISON_BATCH_SIZE}")
    for model in MODELS:
        print(
            f"{model.name:<10} full GD "
            f"SSE={full_results[model.name].loss:.10f}"
        )
        for optimizer in ("sgd", "nesterov", "adam"):
            result = stochastic_results[optimizer][model.name]
            print(f"{'':10} {optimizer:<8} SSE={result.loss:.10f}")
    print()
    print("Figures:")
    for name in (
        "function_minimum.png",
        "model_fits.png",
        "batch_size_comparison.png",
        "optimizer_comparison.png",
    ):
        print(name)


if __name__ == "__main__":
    main()

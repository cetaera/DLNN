"""Parameter counts and softmax derivatives for a self-attention block."""

from decimal import Decimal, localcontext
from typing import Iterable


LOGITS = (-3, 1, 100, 5, -1)
DECIMAL_PRECISION = 100


def self_attention_parameter_count(vector_size: int) -> int:
    """Return the trainable parameter count for values, keys and queries."""
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")
    return 3 * vector_size * (vector_size + 1)


def attention_coefficient_count(sequence_length: int) -> int:
    """Return the number of data-dependent coefficients in attention."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return sequence_length**2


def perceptron_parameter_count(
    vector_size: int,
    sequence_length: int,
    hidden_units: int | None = None,
) -> int:
    """Count parameters in a two-layer fully connected perceptron."""
    if vector_size <= 0 or sequence_length <= 0:
        raise ValueError("vector_size and sequence_length must be positive")

    width = vector_size * sequence_length
    if hidden_units is None:
        hidden_units = width
    if hidden_units <= 0:
        raise ValueError("hidden_units must be positive")

    input_layer = hidden_units * width + hidden_units
    output_layer = width * hidden_units + width
    return input_layer + output_layer


def stable_softmax(values: Iterable[int]) -> tuple[Decimal, ...]:
    """Calculate softmax without overflow and retain very small terms."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        decimal_values = tuple(Decimal(value) for value in values)
        maximum = max(decimal_values)
        exponentials = tuple((value - maximum).exp() for value in decimal_values)
        denominator = sum(exponentials)
        return tuple(value / denominator for value in exponentials)


def softmax_jacobian(
    probabilities: tuple[Decimal, ...],
) -> tuple[tuple[Decimal, ...], ...]:
    """Return the Jacobian with rows indexed by output and columns by logit."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        return tuple(
            tuple(
                probabilities[row]
                * ((Decimal(1) if row == column else Decimal(0))
                   - probabilities[column])
                for column in range(len(probabilities))
            )
            for row in range(len(probabilities))
        )


def main() -> None:
    print("Parameter counts")
    print("Values, keys and queries: 3*D^2 + 3*D = 3*D*(D + 1)")
    print("Attention coefficients: N^2 (computed from the input)")
    print("Two-layer perceptron with H hidden units: 2*H*D*N + H + D*N")
    print("For H = D*N: 2*D^2*N^2 + 2*D*N")
    print()

    probabilities = stable_softmax(LOGITS)
    jacobian = softmax_jacobian(probabilities)

    print(f"z = {LOGITS}")
    print("Softmax values")
    for index, probability in enumerate(probabilities, start=1):
        print(f"y{index} = {probability:.50E}")
    print(f"1 - y3 = {Decimal(1) - probabilities[2]:.16E}")
    print()

    print("Jacobian rows i, columns j: d(y_i)/d(z_j)")
    for row in jacobian:
        print(" ".join(f"{value:.16E}" for value in row))


if __name__ == "__main__":
    main()

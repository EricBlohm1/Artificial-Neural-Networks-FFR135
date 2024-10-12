import numpy as np
from itertools import product

# Perceptron Parameters
eta = 0.05  # Learning rate
epochs = 20  # Number of training epochs

def sgn(b):
    """Activation function: Sign function."""
    return 1 if b >= 0 else -1

def perceptron_train(inputs, targets, n):
    """Train the perceptron to classify a Boolean function."""
    # Initialize weights and threshold
    weights = np.random.normal(0, 1/np.sqrt(n), n)
    threshold = 0

    # Training for the specified number of epochs
    for _ in range(epochs):
        for i, x in enumerate(inputs):
            # Compute the output
            b = np.dot(weights, x) - threshold
            output = sgn(b)

            # Update the weights and threshold
            for j in range(n):
                weights[j] += eta * (targets[i] - output) * x[j]
            threshold -= eta * (targets[i] - output)
    
    return weights, threshold

def perceptron_test(weights, threshold, inputs, targets):
    """Test if the perceptron correctly classifies all input-output pairs."""
    for i, x in enumerate(inputs):
        b = np.dot(weights, x) - threshold
        output = sgn(b)
        if output != targets[i]:
            return False  # Misclassification found
    return True  # All inputs are correctly classified

def generate_boolean_functions(n):
    """Generate all possible Boolean functions for n inputs."""
    num_inputs = 2**n  # Number of input combinations
    inputs = list(product([1, -1], repeat=n))  # Create all input combinations (-1 and 1 instead of 0 and 1)
    
    # There are 2^(2^n) possible Boolean functions, so we need to generate all combinations of outputs.
    boolean_functions = list(product([1, -1], repeat=num_inputs))
    
    return np.array(inputs), boolean_functions

def check_linear_separability(n):
    """Check how many Boolean functions are linearly separable for n dimensions."""
    inputs, boolean_functions = generate_boolean_functions(n)
    separable_count = 0

    # Iterate through all possible Boolean functions
    for targets in boolean_functions:
        weights, threshold = perceptron_train(inputs, targets, n)

        if perceptron_test(weights, threshold, inputs, targets):
            separable_count += 1

    return separable_count

# Check linear separability for n = 2, 3, 4, 5 dimensions
for n in [2, 3, 4, 5]:
    num_separable = check_linear_separability(n)
    total_functions = 2**(2**n)
    print(f"Dimension {n}: {num_separable} out of {total_functions} Boolean functions are linearly separable")

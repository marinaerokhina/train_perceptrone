"""A single sigmoid neuron trained by batch gradient descent.

Educational logistic-regression model, not a multilayer neural network.
The original misspelled class/module names are retained for compatibility.
"""
import numpy as np


class SimplePerceptone:
    def __init__(self, random_state: int = 42):
        rng = np.random.default_rng(random_state)
        self._w = rng.uniform(-0.01, 0.01, size=3)
        self._bias = 0.0
        self._history = []

    def predict(self, samples):
        """Return sigmoid probabilities for a batch of three-feature vectors."""
        x = np.asarray(samples, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("Expected samples with shape (n_samples, 3)")
        logits = np.clip(x @ self._w + self._bias, -500, 500)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_classes(self, samples, threshold: float = 0.5):
        return (self.predict(samples) >= threshold).astype(int)

    def train(self, sample, ans, l_rate: float = 0.5, epochs: int = 5000):
        """Minimize binary cross-entropy via batch gradient descent.

        For sigmoid activation with binary cross-entropy, the derivative
        with respect to logits is (probability - target).
        """
        x = np.asarray(sample, dtype=float)
        y = np.asarray(ans, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("Expected training data with shape (n_samples, 3)")
        if y.shape != (x.shape[0],):
            raise ValueError("Expected one target per training sample")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Targets must be binary: 0 or 1")
        if l_rate <= 0 or epochs <= 0:
            raise ValueError("Learning rate and number of epochs must be positive")

        self._history = []
        for _ in range(epochs):
            p = self.predict(x)
            eps = 1e-12
            loss = -np.mean(y * np.log(np.clip(p, eps, 1 - eps))
                            + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps)))
            self._history.append(float(loss))
            error = p - y
            self._w -= l_rate * (x.T @ error) / x.shape[0]
            self._bias -= l_rate * np.mean(error)
        return self._history

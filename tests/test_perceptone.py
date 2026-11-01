"""Tests for the complete Boolean rule and the sigmoid training example."""
import unittest

import numpy as np

from src.perceptone import SimplePerceptone
from src.start import training_data


class TestSingleNeuron(unittest.TestCase):
    def test_truth_table(self):
        x, y = training_data()
        self.assertEqual(x.shape, (8, 3))
        self.assertEqual(y.tolist(), [0, 0, 0, 0, 0, 1, 1, 1])

    def test_learning_decreases_loss_and_fits_truth_table(self):
        x, y = training_data()
        model = SimplePerceptone(random_state=42)
        history = model.train(x, y, l_rate=0.5, epochs=5000)
        self.assertEqual(len(history), 5000)
        self.assertTrue(np.isfinite(history).all())
        self.assertLess(history[-1], history[0])
        np.testing.assert_array_equal(model.predict_classes(x), y)

    def test_reproducible_training(self):
        x, y = training_data()
        a, b = SimplePerceptone(42), SimplePerceptone(42)
        self.assertEqual(a.train(x, y, 0.5, 20), b.train(x, y, 0.5, 20))

    def test_invalid_inputs(self):
        model = SimplePerceptone()
        with self.assertRaises(ValueError):
            model.predict([[1, 0]])
        with self.assertRaises(ValueError):
            model.train([[1, 0, 1]], [0, 1])
        with self.assertRaises(ValueError):
            model.train([[1, 0, 1]], [2])
        with self.assertRaises(ValueError):
            model.train([[1, 0, 1]], [1], l_rate=0)


if __name__ == "__main__":
    unittest.main()

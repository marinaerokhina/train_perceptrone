import numpy as np


class SimplePerceptone():
    def __init__(self):
        self._w = np.random.rand(3)
        self._bias = np.random.rand(1)

    def predict(self, train):
        res = np.dot(train, self._w)
        result = (res + self._bias)
        return (1 / (1 + np.exp(-result)))

    def train(self, sample, ans, l_rate):
        self._history = []
        sample = np.array(sample)
        ans = np.array(ans)
        for _ in range(10000):
            pred_result = self.predict(sample)
            mse = np.mean((pred_result - ans)**2)
            self._history.append(float(mse))
            err = pred_result - ans
            grad = np.dot(sample.T, err) / len(ans)
            self._w = self._w - (l_rate * grad)
            self._bias = self._bias - (l_rate * np.mean(err))
        return self._history


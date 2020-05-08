import unittest

import numpy as np

from regression.linear import LinearRegression

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 2, 3, 3, 5])
x_test = np.array([11, 12, 13, 14, 15])

m = len(x)
x = x.reshape(m,1)
y = y.reshape(m,1)

class TestLinear(unittest.TestCase):
    def testfit(self):
        LR = LinearRegression(0.0000001, 10, 'Gradient')
        summary = LR.fit(x, y)
        self.assertIsNotNone(summary.get_summary())

        LR = LinearRegression(0.0000001, 10)
        summary = LR.fit(x, y)
        self.assertIsNotNone(summary.get_summary())

    def testloss(self):
        LR = LinearRegression(0.0000001, 10, 'Gradient')
        LR.fit(x, y)
        self.assertIsNotNone(LR.loss(x, y))

    def testpredict(self):
        LR = LinearRegression(0.0000001, 10, 'Gradient')
        LR.fit(x, y)
        self.assertIsNotNone(LR.predict(x_test))

if __name__ == '__main__':
  unittest.main()

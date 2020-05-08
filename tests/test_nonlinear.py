import unittest

import numpy as np

from regression.nonlinear import NonLinearRegression

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 2, 3, 3, 5])
x_test = np.array([11, 12, 13, 14, 15])

m = len(x)
x = x.reshape(m,1)
y = y.reshape(m,1)

class TestNonLinear(unittest.TestCase):
    def testfit(self):
        NR = NonLinearRegression()
        summary = NR.fit(x, y)
        self.assertIsNotNone(summary.get_summary())

    # def testloss(self):
    #     NR = NonLinearRegression()
    #     self.assertIsNotNone(NR.loss(x, y))

    # def testpredict(self):
    #     NR = NonLinearRegression()
    #     self.assertIsNotNone(NR.predict(x))

if __name__ == '__main__':
  unittest.main()

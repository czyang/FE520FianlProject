import unittest

import numpy as np

from regression.stats import Stats

st = Stats()

y_test = np.array([1, 2, 3, 4, 5])
y_pred = np.array([2, 2, 3, 3, 5])

class TestStats(unittest.TestCase):
    def testSSE(self):
        res = st.SSE(y_test, y_pred)
        self.assertEqual(2, res)

    def testMSE(self):
        res = st.MSE(y_test, y_pred)
        self.assertEqual(0.4, res)

    def testRMSE(self):
        res = st.RMSE(y_test, y_pred)
        self.assertAlmostEqual(0.63, res, places=2)

    def testSSR(self):
        res = st.SSR(y_test, y_pred)
        self.assertAlmostEqual(6.0, res, places=2)

    def testSST(self):
        res = st.SST(y_test, y_pred)
        print(res)
        self.assertAlmostEqual(8.0, res, places=2)

    def testR_squared(self):
        res = st.R_squared(y_test, y_pred)
        self.assertAlmostEqual(0.75, res, places=2)

    def testAdj_R_squared(self):
        res = st.Adj_R_squared(y_test, y_pred, 1)
        self.assertAlmostEqual(0.25, res, places=2)

    def testF_statistic(self):
        res = st.F_statistic(y_test, y_pred, 2)
        self.assertAlmostEqual(9.0, res, places=2)

    def testProb(self):
        res = st.Prob(y_test, y_pred, 2)
        self.assertAlmostEqual(0.028, res, places=3)

    def testLog_likelihood(self):
        res = st.Log_likelihood(y_test, y_pred)
        self.assertAlmostEqual(-3.59, res, places=2)

    def testAIC(self):
        res = st.AIC(y_test, y_pred, 3)
        self.assertAlmostEqual(13.19, res, places=2)

    def testBIC(self):
        res = st.BIC(y_test, y_pred, 1, 2)
        self.assertAlmostEqual(7.88, res, places=2)

    def testHQ(self):
        res = st.HQ(y_test, y_pred, 1, 2)
        self.assertAlmostEqual(6.82, res, places=2)

if __name__ == '__main__':
  unittest.main()

import numpy as np
import pandas as pd

class PolynomialFeatures:
  def __init__(self, degree):
    self.degree = degree

  # Get coeffs from input
  def fit(self, X, y=None):
    # Least squares polynomial fit
    return np.polyfit(X, y, deg=self.degree)

  # def tranform(self, X):

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1,   6,  17,  34,  57,  86, 121, 162, 209, 262, 321]
ply = PolynomialFeatures(2)
coeffs = ply.fit(x, y)
# coeffs = np.polyfit(x, y, deg=2)
print("coeffs", coeffs)
print(np.poly1d(coeffs))
yf = np.polyval(np.poly1d(coeffs), x)
print(yf)

print('%.1g' % max(y - yf))
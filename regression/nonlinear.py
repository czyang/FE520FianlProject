import numpy as np
from regression.linear import LinearRegression

class NonLinearRegression(LinearRegression):
	def __init__(self, learningrate = 0.0000001, max_iter = 10, degree = 2):
		LinearRegression.__init__(self, learningrate, max_iter = 10, method = 'Normal')
		self.degree = degree

	def enlarge(self, x):
		res = []
		for i in range(0, len(x)):
			slices = []
			for j in range(1, self.degree + 1):
				slices.append(x[i] ** j)
			res.append(np.array(slices))
		res = np.array(res)
		return res.reshape((len(x), self.degree))	

	def fit(self, x_train, y_train):
		x_train = self.enlarge(x_train)
		summary = LinearRegression.fit(self, x_train, y_train)
		summary.setTitle("Nonlinear Regression")
		return summary

	def loss(self, x_train, y_train):
		y_pred = self.predict(x_train)
		return np.mean((y_train - y_pred) ** 2)

	def predict(self, x_test, enlarged = True):
		if enlarged == False:
			x_test = self.enlarge(x_test)
		return LinearRegression.predict(self, x_test)
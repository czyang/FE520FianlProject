import numpy as np
# import statistic as s

class LinearRegression():
	def __init__(self, learningrate, max_iter):
		self.learningrate = learningrate
		self.coef = 0
		self.intercept = 0
		self.max_iter = max_iter

	def fit(self, x_train, y_train):
		self.coef = np.zeros((len(x_train[0])))
		for i in range(0, self.max_iter):
			d_coef = np.mean((x_train * self.coef + self.intercept - y_train) * x_train)
			d_intercept = np.mean((x_train * self.coef + self.intercept - y_train))

			self.coef = self.coef - self.learningrate * d_coef
			self.intercept = self.intercept - self.learningrate * d_intercept

			self.loss(x_train, y_train)

	def loss(self, x_train, y_train):
		y_pred = self.predict(x_train)
		return np.mean((y_train - y_pred) ** 2)

	def predict(self, x_test):
		y_pred = x_test * self.coef + self.intercept
		return y_pred
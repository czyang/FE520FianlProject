import numpy as np

from regression.stats import Stats
from regression.summary import Summary


class LinearRegression():
	def __init__(self, learningrate = 0.0000001, max_iter = 10, method = 'Normal'):
		self.learningrate = learningrate
		self.coef = []
		self.intercept = 0
		self.max_iter = max_iter
		self.method = method

	def fit(self, x_train, y_train):
		if self.method == 'Gradient':
			self.coef = np.zeros((len(x_train[0])))
			num = len(x_train)
			for i in range(0, self.max_iter):
				print("epoch: ", i + 1)
				d_coef = np.zeros((len(x_train[0])))
				d_intercept = 0.0
				for j in range(0, num):
					d_coef += ((np.dot(x_train[j], self.coef) + self.intercept - y_train[j]) * x_train[j] / num)
					d_intercept += ((np.dot(x_train[j], self.coef) + self.intercept - y_train[j]) / num)

			# using directly numpy calculation may face data out of range under some situations
			# for i in range(0, self.max_iter):
			# 	d_coef = np.mean((x_train * self.coef + self.intercept - y_train) * x_train)
			# 	d_intercept = np.mean((x_train * self.coef + self.intercept - y_train))

				self.coef = self.coef - self.learningrate * d_coef
				self.intercept = self.intercept - self.learningrate * d_intercept

			# self.loss(x_train, y_train)
		elif self.method == 'Normal':
			x0 = [0.1 for i in x_train]
			x_matrix = np.mat(x_train)
			y_matrix = np.mat(y_train)
			x0_matrix = np.mat(x0)
			x0_T = x0_matrix.T
			x_T = x_matrix
			x_T = np.hstack((x0_T, x_T))
			
			x_matrix = x_T.T
			tmp1 = np.dot(x_matrix, x_T)
			tmp2 = np.linalg.inv(tmp1)
			theta = tmp2 * x_matrix * y_matrix
			theta = np.array(theta)
			for i in range(1, len(theta)):
				self.coef.append(theta[i][0])
			self.intercept = [theta[0][0]]

		stats = Stats()

		summary = Summary()
		summary.setTitle("Linear Regression")
		summary.append("Model:", "Regression")
		summary.append("Method:", "Linear Regression")
		summary.appendDate()
		summary.appendTime()
		# TODO: Add stats here

		return summary

	def loss(self, x_train, y_train):
		y_pred = self.predict(x_train)
		return np.mean((y_train - y_pred) ** 2)

	def predict(self, x_test):
		y_pred = []

		for i in range(0, len(x_test)):
			y_pred.append(np.dot(x_test[i], self.coef) + self.intercept)

		y_pred = np.array(y_pred)
		y_pred = y_pred.reshape((len(x_test), 1))
		return y_pred
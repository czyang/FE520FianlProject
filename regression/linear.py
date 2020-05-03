import numpy as np
# import statistic as s

class LinearRegression():
	def __init__(self, learningrate, max_iter = 10):
		self.learningrate = learningrate
		self.coef = 0
		self.intercept = 0
		self.max_iter = max_iter

	def fit(self, x_train, y_train):
		self.coef = np.zeros((len(x_train[0])))
		num = len(x_train)
		for i in range(0, self.max_iter):
			print("epoch: ", i + 1)
			d_coef = np.zeros((len(x_train[0])))
			d_intercept = 0.0
			for j in range(0, num):
				d_coef += (np.dot(x_train[j], self.coef) + self.intercept - y_train[j]) * x_train[j] / num
				d_intercept += (np.dot(x_train[j], self.coef) + self.intercept - y_train[j]) / num

		# using directly numpy calculation may face data out of range under some situations
		# for i in range(0, self.max_iter):
		# 	d_coef = np.mean((x_train * self.coef + self.intercept - y_train) * x_train)
		# 	d_intercept = np.mean((x_train * self.coef + self.intercept - y_train))

			self.coef = self.coef - self.learningrate * d_coef
			self.intercept = self.intercept - self.learningrate * d_intercept

			# self.loss(x_train, y_train)

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
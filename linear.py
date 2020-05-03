import numpy as np
# import statistic as s

class LinearRegression():
	def __init__(self, learningrate = 0.0000001, max_iter = 10, method = 'Normal'):
		self.learningrate = learningrate
		self.coef = 0
		self.intercept = 0
		self.max_iter = max_iter
		self.method = method

	def fit(self, x_train, y_train):
		if self.method == 'Gradient':
			m = len(x_train)
			x_train = x_train.reshape(m, len(x_train[0]))
			y_train = y_train.reshape(m, len(x_train[0]))
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
			x_T = x_matrix.T
			x_T = np.hstack((x0_T, x_T))
			
			x_matrix = x_T.T
			tmp1 = np.dot(x_matrix, x_T)
			tmp2 = np.linalg.inv(tmp1)
			print(tmp2)
			print(x_matrix)
			print(y_matrix.T)
			theta = tmp2 * x_matrix * y_matrix.T
			
			self.coef = theta[1]
			self.intercept = theta[0]


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

import csv

with open('train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    y = np.array([int(row['price']) for row in reader])

with open('train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    x = np.array([int(row['sqft_living']) for row in reader])

with open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    yt = np.array([int(row['price']) for row in reader])

with open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    xt = np.array([int(row['sqft_living']) for row in reader])


csvfile.close()
n = len(xt)
xt = xt.reshape(n,1)
yt = yt.reshape(n,1)
LR = LinearRegression(method = 'Normal')
LR.fit(x, y)
print(LR.coef, LR.coef.shape, LR.intercept)
y_pred = LR.predict(xt)
import matplotlib.pyplot as plt

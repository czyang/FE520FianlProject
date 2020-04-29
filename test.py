import numpy as np
import csv
import statistic as s

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
m = len(x)
x = x.reshape(m,1)
y = y.reshape(m,1)
csvfile.close()

with open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    yt = np.array([int(row['price']) for row in reader])

with open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    xt = np.array([int(row['sqft_living']) for row in reader])

n = len(xt)
xt = xt.reshape(n,1)
yt = yt.reshape(n,1)
csvfile.close()

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

LR = LinearRegression(0.0000001, 100)
LR.fit(x, y)
print(LR.coef, LR.intercept)
y_pred = LR.predict(xt)
# print(y_pred)
print(s.Log_likelihood(yt, y_pred))
print(s.AIC(yt, y_pred, 2))
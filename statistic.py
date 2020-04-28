import numpy as np
import math

def SSE(y_test, y_pred):
	return np.sum((y_test - y_pred) ** 2)

def MSE(y_test, y_pred):
	return SSE(y_test, y_pred) / len(y_test)

def RMSE(y_test, y_pred):
	return math.sqrt(MSE(y_test, y_pred))

def SSR(y_test, y_pred):
	mean = np.mean(y_test)
	return np.sum((mean - y_pred) ** 2)

def SST(y_test, y_pred):
	return SSE(y_test, y_pred) + SSR(y_test, y_pred)

def R_squared(y_test, y_pred):
	return SSR(y_test, y_pred) / SST(y_test, y_pred)

def Adj_R_squared(y_test, y_pred, p):
	n = len(y_test)
	return 1 - (n - 1) * (R_squared(y_test, y_pred) ** 2) / (n - p - 1)

def F_statistic(y_test, y_pred):
	pass

def Prob(y_test, y_pred):
	pass

def Log_likelihood():
	pass

def AIC():
	pass

def BIC():
	pass

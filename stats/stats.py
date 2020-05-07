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

def F_statistic(y_test, y_pred,p):
	n = len(y_test)
	MSM = SSR(y_test, y_pred) / (p - 1)
	MSE = SSE(y_test, y_pred) / (n - p)
	return MSM/MSE

def Prob(y_test, y_pred):
	import scipy.stats as stats
	df1 = len(y_test) - 1
	df2 = len(y_pred) - 1
	p_value = stats.f.sf(F, df1, df2)
	return p_value


def Log_likelihood(y_test, y_pred):
	s = 0
	e = 2.
	for i in range(0, len(y_test)):
		s += math.log(1 / np.sqrt(2 * math.pi)) + (y_test[i] - y_pred[i]) **2 / 2
	return s

def AIC(y_test, y_pred, p):
	return -2 * Log_likelihood(y_test, y_pred) + 2 * p

def BIC(y_test, y_pred, p, n): # n sample numbers, p number of independent variables x1,x2,..b 
	return -2 * Log_likelihood(y_test, y_pred) + math.log(n) * p

def HQ(y_test, y_pred, p, n): # n sample numbers, p number of independent variables x1,x2,..b 
	return -2 * Log_likelihood(y_test, y_pred) + math.log(math.log(n)) * p
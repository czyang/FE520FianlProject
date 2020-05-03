import numpy as np
import csv
import sys
sys.path.append("..")
import linear
sys.path.append("../../stats")
import stats as s

with open('../../train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    y = np.array([int(row['price']) for row in reader])

with open('../../train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    x = np.array([int(row['sqft_living']) for row in reader])

with open('../../train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    x1 = np.array([int(row['sqft_above']) for row in reader])

with open('../../test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    yt = np.array([int(row['price']) for row in reader])

with open('../../test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    xt = np.array([int(row['sqft_living']) for row in reader])

with open('../../test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    xt1 = np.array([int(row['sqft_above']) for row in reader])

m = len(x)
# x = np.hstack([x, x1])
x = x.reshape(m,1)
x1 = x1.reshape(m,1)
y = y.reshape(m,1)
x = np.hstack([x, x1])
print(x)
csvfile.close()

n = len(xt)
xt = xt.reshape(n,1)
xt1 = xt1.reshape(n,1)
yt = yt.reshape(n,1)
xt = np.hstack([xt, xt1])
print(xt)
csvfile.close()
LR = linear.LinearRegression(0.0000001, 10)
LR.fit(x, y)
print(LR.coef, LR.intercept)
y_pred = LR.predict(xt)
print(y_pred)
print(s.Log_likelihood(yt, y_pred))
print(s.AIC(yt, y_pred, 2))
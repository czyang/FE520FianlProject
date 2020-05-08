import numpy as np
import csv
# import sys
from sklearn import linear_model
# sys.path.append("..")

# sys.path.append("../stats")
# import stats as s
from regression import linear

relPath = ''

with open(relPath + 'train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    y = np.array([int(row['price']) for row in reader])

with open(relPath + 'train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    x = np.array([int(row['sqft_living']) for row in reader])

with open(relPath + 'train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    x1 = np.array([int(row['sqft_above']) for row in reader])

with open(relPath + 'test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    yt = np.array([int(row['price']) for row in reader])

with open(relPath + 'test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    xt = np.array([int(row['sqft_living']) for row in reader])

with open(relPath + 'test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    xt1 = np.array([int(row['sqft_above']) for row in reader])

m = len(x)
x_ = x
y_ = y
x = x.reshape(m,1)
x1 = x1.reshape(m,1)
y = y.reshape(m,1)
xm = np.hstack([x, x1])
# print(x)
csvfile.close()

n = len(xt)
xt = xt.reshape(n,1)
xt1 = xt1.reshape(n,1)
yt = yt.reshape(n,1)
xt = np.hstack([xt, xt1])
# print(xt)
csvfile.close()

print("One dimension LinearRegression using gradient descent")
LR = linear.LinearRegression(0.0000001, 10, 'Gradient')
LR.fit(x, y)
print(LR.coef, LR.intercept)

print("Two dimension LinearRegression using normal equation")
LZ = linear.LinearRegression(0.0000001, 10)
sm = LZ.fit(xm, y)
print(LZ.coef, LZ.intercept)
print(sm.get_summary())

print("One dimension LinearRegression using normal equation")
LF = linear.LinearRegression(0.0000001, 10)
LF.fit(x, y)
x1 = np.linspace(0, 14000)
y1 = LF.coef * x1 + LF.intercept
print(LF.coef, LF.intercept)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x1, y1)
plt.plot(x, y, 'ro')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.title('Normal Equation 1 feature')
plt.show()

print("One dimension LinearRegression using sklearn")
model = linear_model.LinearRegression()
model.fit(x, y)
# x1 = np.linspace(0, 14000)
print(model.coef_, model.intercept_)
y1 = model.coef_ * x1 + model.intercept_
plt.figure()
plt.plot(x1,y1[0])
plt.plot(x,y, 'ro')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.title('benchmark')
plt.show()
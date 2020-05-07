import numpy as np
import csv
import sys
import nonlinear
sys.path.append("../stats")
import stats as s

with open('../../train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    y = np.array([int(row['price']) for row in reader])

with open('../../train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    x = np.array([int(row['sqft_living']) for row in reader])

with open('../../test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    yt = np.array([int(row['price']) for row in reader])

with open('../../test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    xt = np.array([int(row['sqft_living']) for row in reader])

m = len(x)
# x = np.hstack([x, x1])
x = x.reshape(m,1)
y = y.reshape(m,1)

csvfile.close()

n = len(xt)
xt = xt.reshape(n,1)
yt = yt.reshape(n,1)

csvfile.close()
import matplotlib.pyplot as plt
LR = nonlinear.NonLinearRegression()
LR.fit(x, y)
print(LR.coef, LR.intercept)
y_pred = LR.predict(xt)
x1 = np.linspace(0, 140000)
y1 = LR.predict(x1)

plt.figure()
plt.plot(x1,y1)
plt.plot(x,y, 'ro')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.title('Gradient Descent')
plt.show()
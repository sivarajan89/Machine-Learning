import numpy as np
import matplotlib.pyplot as plt

# Training Data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 3, 5, 8, 10, 13, 14, 18, 18, 21])

n = np.size(x)
mx, my = np.mean(x), np.mean(y)
crossDevXY = np.sum(y * x) - n * mx * my
devX = np.sum(x * x) - n * mx * mx
beta1 = crossDevXY / devX
beta0 = my - beta1 * mx

xm = np.vstack((np.ones((n)), x))
beta = np.array([beta0, beta1])

iter = 10
alpha = 0.1
rmse = 0
loop = 1
while loop:
    rmse_old = rmse
    y_pred = np.dot(beta, xm)
    rmse = np.sum((y_pred-y)**2)/(2*n)
    print(rmse)
    beta = beta - (alpha * np.mean((y_pred-y)*x))*np.ones(2)
    if rmse==rmse_old:
        loop = 0

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.show()
print("Estimated Coefficient: " + str(beta))
print("RMSE:" + str(rmse))

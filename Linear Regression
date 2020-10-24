import numpy as np
import matplotlib.pyplot as plt

# Training Data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 3, 5, 8, 10, 13, 14, 18, 18, 21])

# Linear Regression Model
n = np.size(x)
mx, my = np.mean(x), np.mean(y)
crossDevXY = np.sum(y * x) - n * mx * my
devX = np.sum(x * x) - n * mx * mx
beta1 = crossDevXY / devX
beta0 = my - beta1 * mx
print("Estimated Coefficient: " + str(beta0) + " " + str(beta1))

# Plotting Regression Line
plt.scatter(x, y, color="m", marker="o", s=30)
y_pred = beta0 + beta1 * x
plt.plot(x, y_pred, color="g")
plt.show()

# Mean Square Error
rmse = np.sum((y_pred-y)**2)/(2*n)
print("RMSE: " + str(rmse))

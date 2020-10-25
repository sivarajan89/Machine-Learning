import numpy as np
import matplotlib.pyplot as plt

a = np.transpose(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.1, 0.3, 0.32, 0.12, 0.2, 0.17, 0.24, 0.13, 0.33, 0.19],
                           [21, 43, 10, 51, 18, 12, 19, 31, 51, 11]]))
b = np.transpose(np.array([0.1341, 0.4153, 0.5691, 0.2315, 0.2315, 0.3216, 0.9253, 0.1943, 0.2281, 0.1529]))

s = np.shape(a)
p = s[0]
n = s[1]

beta = np.transpose(np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(a), a)), np.transpose(a)), b))
b_pred = np.dot(a, beta)
print(b_pred)
print(a[:, 0])

plt.scatter(a[:, 0], b)
plt.plot(a[:, 0], b_pred)
plt.show()

# Mean Square Error
mse = np.mean((b - b_pred)**2)
print("Mean Square Error is " + str(mse))

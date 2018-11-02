import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def computeCost(X, y, theta):
    return ((X.dot(theta) - y)**2).sum() / (2*len(X))

def pradientDescent(X, y, theta, alpha, iterations):
    for i in range(iterations):
        k = np.dot(X.T, (X.dot(theta) - y)) / len(X)
        theta = theta - alpha * k
    return theta


# --------Part 2 : Plotting
print('Plotting Data ...')
data = np.loadtxt('ex1/ex1data1.txt', delimiter=',')
X = data[:, 0:1]
Y = data[:, 1:2]
m = len(data)
plt.scatter(X, Y, c='red', marker='x', linewidths=4)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
# plt.show()
print('Plotting Data end ...')

# --------Part 3 : Cost and Gradient descent
X = np.c_[np.ones((m, 1)), X]
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

# Test J is OK
print('Testing the cost function...')
J = computeCost(X, Y, theta)
print('With theta = [0, 0]\nCost compute = {}'.format(J))
print('Expect cost value (approx) 32.07')
J = computeCost(X, Y, np.array([[-1], [2]]))
print('With theta = [-1, 2]\nCost compute = {}'.format(J))
print('Expect cost value (approx) 54.24')
print('Testing the cost function end...')

# run gradien descent
theta = pradientDescent(X, Y, theta, alpha, iterations)


# print theta
print('Theta found by gradient descent:\n {}'.format(theta))
print('Expected theta values: -3.6303  1.1664')

# Plot the linear fit
plt.plot(X[:, 1], X.dot(theta))
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]) * theta
print('For population = 35,000, we predict a profit of {}'.format(predict1*1000))
predict2 = np.array([1, 7]) * theta
print('For population = 70,000, we predict a profit of {}'.format(predict2*1000))
print('Gradient descent end ...')

# --------Part 4 : Visualizing J(theta_0, theta_1)
print('Visualizing J(theta_0, theta_1) ...')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.array([computeCost(X, Y, np.array([[theta0], [theta1]])) for theta0 in theta0_vals for theta1 in theta1_vals]).reshape(100, 100)
J_vals = J_vals.T

xzim, yzim = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xzim, yzim, J_vals, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, linprog
import matplotlib.pyplot as plt
from scipy import linalg


def cost_least_squares(params, X, y):
    return np.sum((y - X.dot(params)) ** 2) / float(np.size(y))


def cost_ridge(params, X, y, lamb):
    return (lamb * np.sum(params**2) + np.sum((y - X.dot(params)) ** 2)) / float(np.size(y))


def cost_lasso(params, X, y, lamb):
    return (lamb * np.sum(np.abs(params)) + np.sum((y - X.dot(params)) ** 2)) / float(np.size(y))


def regression_plot(title, x_pred, y_pred, x_org, y_org, ylim=(-2, 2), xlim=(-3, 3)):
    """
    Plots the underlying regression.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x_org, y_org, '.', color='green', alpha=1)
    plt.plot(x_pred, y_pred, '.', color='red', alpha=1)
    plt.title(title)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.show()


## -- Data Generating Process -- ##
X_ = np.random.normal(0, 1.5, 400)
y = [i + np.random.normal(0, .4) for i in np.sin(X_)]


### --- OLS Regression with 4 polynomial features --- ###
poly = PolynomialFeatures(4)
X = poly.fit_transform(X_.reshape(-1, 1))

theta = np.array([5.0] * len(X[0]))
output = minimize(cost_least_squares, theta, args=(X, y), method='SLSQP')

yhat = X @ output.x

regression_plot('OLS: 4 Polynomial Features', X_, yhat, X_, y)


### --- Ridge Regression with 4 polynomial features --- ###
poly = PolynomialFeatures(4)
X = poly.fit_transform(X_.reshape(-1, 1))

theta = np.array([5.0] * len(X[0]))
pconv = []
ran = np.linspace(0, 1000, 25)
for i in ran:
    output = minimize(cost_ridge, theta, args=(X, y, i), method='SLSQP')
    pconv.append(output.x)
    yhat = X @ output.x
    plt.plot(sorted(X_), sorted(yhat), color='red', alpha=1)
plt.title('Ridge Regression with increasing regularization.')
plt.plot(X_, y, '.', color='green', alpha=1)
plt.ylim((-2, 2))
plt.xlim((-3, 3))
plt.show()

plt.title('Ridge Regression Parameter Shrinkage')
plt.plot(pd.DataFrame(pconv, columns=['x1', 'x2', 'x3', 'x4', 'x5'], index=ran))
plt.legend()
plt.show()


### --- Ridge Regression with 4 polynomial features --- ###
poly = PolynomialFeatures(4)
X = poly.fit_transform(X_.reshape(-1, 1))

theta = np.array([5.0] * len(X[0]))

pconv = []
for i in ran:
    output = minimize(cost_lasso, theta, args=(X, y, i), method='SLSQP')
    pconv.append(output.x)
    yhat = X @ output.x
    plt.plot(sorted(X_), sorted(yhat), color='red', alpha=1)
plt.plot(X_, y, '.', color='green', alpha=1)
plt.ylim((-2, 2))
plt.xlim((-3, 3))
plt.show()

plt.title('Lasso Regression Parameter Shrinkage')
plt.plot(pd.DataFrame(pconv, columns=['x1', 'x2', 'x3', 'x4', 'x5'], index=ran))
plt.legend()
plt.show()


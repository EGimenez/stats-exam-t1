from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression as OLS
from sklearn.preprocessing import normalize

# Problem setup
X, Y, coef = make_regression(n_samples=90, n_features=100, n_informative=5, n_targets=1, random_state=1880, coef=True)

#X_norm = normalize(X)
X_norm = X

kf = KFold(n_splits=10)
kf.get_n_splits(X_norm)

# Lasso
lasso = Lasso(random_state=0, max_iter=3000000)
alphas = [0.000007, 0.00002, 0.00004, 0.00005, 0.00008, 0.0001, 0.00012, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.002]

errors = list(np.zeros((len(alphas), 1)))

def f(alpha):
	error = 0
	lasso = Lasso(random_state=0, max_iter=3000000, alpha=alpha)

	for train_index, test_index in kf.split(X_norm):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		lasso.fit(X_train, Y_train)
		x_index = lasso.coef_ != 0

		X_train_OLS = X_train[:, x_index]
		X_test_OLS = X_test[:, x_index]

		error += mean_squared_error(Y_test, OLS().fit(X_train_OLS, Y_train).predict(X_test_OLS))

	return error

def get_OLS(alpha):
	lasso = Lasso(random_state=0, max_iter=3000000, alpha=alpha)
	lasso.fit(X_norm, Y)
	x_index = lasso.coef_ != 0
	X_OLS = X_norm[:, x_index]

	return OLS().fit(X_OLS, Y)


f(0.1)
ols = get_OLS(0.1)
print('Hey ho lets go')
print(ols.coef_)
print(coef[coef !=0])



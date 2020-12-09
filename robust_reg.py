import matplotlib.pyplot as plt
from scipy import stats
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math

np.random.seed(42)
n= 400

#normal
ne = np.random.normal(0,0.1)
#student
te = np.random.standard_t(n-1)

#Train data
X = np.random.normal(size=n).tolist()
y = [np.sin(i) + te for i in X]

df = pd.DataFrame({'x': X, 'y': y})
df['x2'] = df.x.apply(lambda x : x**2)
df['x3'] = df.x.apply(lambda x : x**3)

# Fit the model with order 3
results = smf.ols(formula='y ~ x + x2 + x3', data=df).fit()
print(results.params)

#plot

plt.plot(df.x, df.y, 'o', color='black')
y = results.params[0] + df.x*results.params[1] + df.x2*results.params[2] + df.x3*results.params[3]
df['y2'] = y
plt.plot(df.x, df.y2, 'o', color='blue')
plt.show()



# Add outliers
lis = [-1.5, -1.55, -1.4, -1.45, -1.6]
for i in lis:
    point = {'x': i, 'x2': i**2, 'x3': i**3, 'y': 1 + np.random.normal(0,0.2)}
    df = df.append(point, ignore_index=True)

# for i in [-i for i in lis]:
#     point = {'x': i, 'x2': i ** 2, 'x3': i ** 3, 'y': -1 + np.random.normal(0, 0.2)}
#     df = df.append(point, ignore_index=True)

# get its leverage
reg = smf.ols(formula='y ~ x + x2 + x3', data=df).fit()
influence = reg.get_influence()
leverage = pd.Series(influence.hat_matrix_diag)[n]
print(leverage)

#Print prediction now
plt.plot(df.x, df.y, 'o', color='black')
y = reg.params[0] + df.x*reg.params[1] + df.x2*reg.params[2] + df.x3*reg.params[3]
df['y2'] = y
plt.plot(df.x, df.y2, 'o', color='blue')
plt.show()

# Robust regression
df.x_ = sm.add_constant(df[['x', 'x2', 'x3']])
rlm_model = sm.RLM(df.y, df.x_, M=sm.robust.norms.HuberT())
rlm = rlm_model.fit()
rlm.params

#Plot
plt.plot(df.x, df.y, 'o', color='black')
y = rlm.params[0] + df.x*rlm.params[1] + df.x2*rlm.params[2] + df.x3*rlm.params[3]
df['y2'] = y
plt.plot(df.x, df.y2, 'o', color='blue')
plt.show()


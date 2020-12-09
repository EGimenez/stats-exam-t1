import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


np.random.seed(42)
n = 10

#Train data
salary = np.random.normal(1500, 200, size=n).tolist()
age = [np.random.randint(18,65) for i in range(n)]
y = [1.4 * salary[i] + 0.2*age[i] + np.random.normal(0,0.1) for i in range(n)]

df = pd.DataFrame({'y':y, 'salary': salary, 'age': age})

#Fit linear model correctly specified
results = smf.ols(formula='y ~ salary + age', data=df).fit()
print(results.params)
# as n gets large the values of coef get to be the same


#Fit linear model over parameterized
income = np.random.normal(10000, 2000, size=n).tolist()
df['income'] = income

results = smf.ols(formula='y ~ salary + age + income', data=df).fit()
print(results.params)
# still get real values


#Fit linear model over parameterized

results = smf.ols(formula='y ~ salary ', data=df).fit()
print(results.params)

# still get real values
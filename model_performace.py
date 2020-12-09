import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.random import choice
from scipy.stats.distributions import chi2

np.random.seed(42)
n = 1000

# Train data
salary = (np.random.normal(1500, 200, size=n) + np.random.normal(150, 50, size=n)).tolist()
age = [np.random.randint(18, 65) for i in range(n)]
y = [1.4 * salary[i] + 0.2 * age[i] + np.random.normal(0, 0.1) for i in range(n)]

df = pd.DataFrame({'y': y, 'salary': salary, 'age': age})

# Define 2 models, a shorter one a longer one

# Short and true
true = smf.ols(formula='y ~ salary + age', data=df).fit()
true_ll = true.llf


# Fit linear model over parameterized
def get_sex(x):
    if x >= 1500:
        return choice([0, 1], p=[0.8, 0.2])
    else:
        return choice([0, 1], p=[0.2, 0.8])


df['sex'] = df.salary.apply(lambda x: get_sex(x))
df['non_sense'] = [np.random.normal(100,80) for i in range(n)]

long = smf.ols(formula='y ~ salary + age + sex + non_sense', data=df).fit()
long_ll = long.llf

# We can see that the longer one has higher log likelihood lets test it. Non significantly good
def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))

LR = likelihood_ratio(true_ll,long_ll)
p = chi2.sf(LR, 1) # L2 has 1 DoF more than L1
p

# Lets see the MSE
in_sample_mse_true = sum(true.resid**2)
in_sample_mse_long = sum(true.resid**2)

in_sample_mse_true < in_sample_mse_long


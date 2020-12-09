import random
import numpy as np
import scipy
import scipy.stats as prob
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as stats
from scipy import stats
from statsmodels.stats.power import TTestIndPower, tt_ind_solve_power

# Gen data

random.seed(500) #for results to be recreated
N = 1000 #number of samples to take from each population
a = [random.gauss(0,1) for x in range(N)] #take N samples from population A
b = [random.gauss(5,1) for x in range(N)] #take N samples from population B

#Plot
# sns.kdeplot(a, shade=True)
# sns.kdeplot(b, shade=True)
# plt.title("Independent Sample T-Test")

#See stat and power
h0 = 0
h1 = 5

tStat, pValue = scipy.stats.ttest_1samp(a, h1, axis=0)
print(tStat*(-1))
#print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) #print the P-Value and the T-Statistic

print(prob.norm(5, 20).cdf(tStat*(-1))) #power


effect_size= 5
sample_size=100
alpha=.05
ratio=1.0
statistical_power = tt_ind_solve_power(effect_size=effect_size, nobs1=sample_size, alpha=alpha, ratio=1.0, alternative='two-sided')
print("The statistical power is: {0}".format(statistical_power))

# Compute sampple size to get the power that we want play with this values in solve_pwer

from statsmodels.stats.power import TTestIndPower
# parameters for power analysis
sample_size = 100
alpha = 0.05
power = 0.8
ratio_ =1

analysis = TTestIndPower()
effect_size = analysis.solve_power(power=power, nobs1=sample_size, ratio=ratio_, alpha=alpha)
print('Eff Size: %.3f' % effect_size)
###Sample Size: 16.442
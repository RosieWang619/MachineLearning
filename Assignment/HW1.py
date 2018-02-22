
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[2]:

np.random.seed(6996)
Epsilon = np.random.randn(500)
X = np.random.normal(0, 2, (500, 500))  # this can also generate a 500*500 matrix of random numbers of normal distribution
# same as:
# X = np.random.normal(0,2,500*500)
# X = np.reshape(X, (500,500))
X = np.matrix(X)
slopesSet = np.random.uniform(1, 5, 500)
# Y = sapply(2:500,function(z) 1+X[,1:z]%*%slopesSet[1:z]+Epsilon)


# In[3]:

Y = np.array(list(map(lambda i: 1 + np.inner(X[:, :i + 1], slopesSet[:i + 1]) + Epsilon, range(500))))  # main function of Y
# since list(map()) gives us a list of n*1 matrix of list
# we have to transfer them into array(matrix) without '[]'
# and delete the useless first column
Y = np.array(list(chain(*Y)))
Y = Y[1:].transpose()
Y.shape


# In[4]:

X_train = X[:, 0:490]
Y_train = Y[:, 489]
X_train = sm.add_constant(X_train)
lm1 = sm.OLS(Y_train, X_train).fit()

fig, ax = plt.subplots(figsize=(10, 8))  # creating fig and subplots with size
ax.set_title("Coefficients's P-Values for 490 Predictors", fontsize=15)
ax.set_ylabel('P-Value')
ax.set_xlabel('Coefficient')
plt.plot(range(0, 490), lm1.pvalues[0:490], 'o')
plt.ylim(0, 0.08)
plt.xlim(0, 500)
plt.show()


# In[5]:

lm1.summary()
# By looking at the summary of the regression, we figure out the Beta_1's confidence interval is in the first row
# And the confidence interval is 95% by default


# In[8]:

r_squared = []
conf_lower = []
conf_upper = []

for i in range(2, 500):
    X_train = sm.add_constant(X[:, 0:i])
    Y_train = Y[:, i - 1]
    lm = sm.OLS(Y_train, X_train).fit()
    r_squared.append(lm.rsquared)
    conf_lower.append(lm.conf_int()[0, 0])
    conf_upper.append(lm.conf_int()[0, 1])


# In[9]:

fig, ax = plt.subplots(figsize=(10, 8))  # creating fig and subplots with size
ax.set_title("Improvement of Fit with Number of Predictors", fontsize=15)
ax.set_ylabel('Determination Coefficient')
ax.set_xlabel('Number of Predictors')
plt.plot(r_squared)
plt.show()


# In[14]:

fig, ax = plt.subplots(figsize=(18, 12))  # creating fig and subplots with size
ax.set_title("Confidence Intervals for Beta_1", fontsize=20)
ax.set_ylabel('95% Confidence Intervals', fontsize=15)
ax.set_xlabel('Number of Predictors', fontsize=15)
plt.plot(conf_lower)
plt.plot(conf_upper)
plt.ylim(-20, 30)
plt.xlim(0, 500)
plt.show()


# In[ ]:

from bit import samplers
import numpy as np
import matplotlib.pyplot as plt
import corner
import tqdm

""" 
Basic example for fitting fake data with a linear model using emcee (all hard coded in samplers.py)
Future improvements could be to dynamically allow users to specify models and turning the samplers into a class containing more methods which users can select
Also adding functionality to get the uncertainties on the fitted parameters - which is simply the 1 sigma credible intervals of the flattened samples returned by samplers.fit()
Everything is more or less hard coded. We need to make the samplers functions more flexible.
"""

# model is NOT user defined here. it is hard coded for the first version of this package
model = samplers.linear_model

# fake data with noise
m_true = 2.5
c_true = 1.0
x = np.linspace(0, 10, 50)
yerr = 0.5
line = model([m_true, c_true], x)
y = line + np.random.normal(0, yerr, len(x))

# plot fake data
fig, axes = plt.subplots(1, figsize=(7, 7))
axes.errorbar(x=x, y=y, yerr=yerr, fmt='.',
              label='Data with noise from the linear model')
axes.plot(x, line, label='Linear model: \nm_true = {}, c_true = {}'.format(
    m_true, c_true))
axes.plot()
plt.legend()
plt.show()

# fit the data with our function (soon to be callable from a python package)
# Our function does everything under the hood which is easy for users
samples = samplers.fit(samplers.log_prob, initial_guess=[
                       1.0, 1.0], args=(x, y, yerr))

# print results
res = samples.mean(axis=0)
m_fit, c_fit = [round(x, 2) for x in [res[0], res[1]]]
print(m_fit, c_fit)

# plot the results
fig = corner.corner(
    samples, labels=['m_fit', 'c_fit'], truths=[m_true, c_true])

# plot the data with our fit values over the true value
fig, axes = plt.subplots(1, figsize=(7, 7))
axes.errorbar(x=x, y=y, yerr=yerr, fmt='.',
              label='Data with noise from the linear model')
axes.plot(x, line, label='Linear model: \nm_true = {}, c_true = {}'.format(
    m_true, c_true), lw=5, c='red')
axes.plot(x, model([m_fit, c_fit], x), c='green',
          label='Fitted model: \nm_fit = {}, c_fit = {}'.format(m_fit, c_fit))
plt.legend()
plt.show()

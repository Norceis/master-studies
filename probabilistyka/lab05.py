import pymc3 as pm
import numpy


with pm.Model() as model:
    x = pm.Normal('x', mu=1, sigma=5)


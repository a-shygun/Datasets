from ISLP import load_data
from ISLP.models import ModelSpec
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = load_data('Auto').reset_index() # fixes the fact that the names were acting as indexes. now its a seperate column
print(df.head())
X = sm.add_constant(df['horsepower'])  # adds intercept 1
y = df['mpg']

model = sm.OLS(y, X).fit()
    
print(model.summary())
print(model.params)
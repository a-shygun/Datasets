import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../Data/Diabetes.csv')

df = df.drop('Outcome', axis=1)

X = df.drop('DiabetesPedigreeFunction', axis=1)
y = df[['DiabetesPedigreeFunction']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# reg = LazyRegressor(verbose=0, ignore_warnings=True)
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)
# print(models)

reg = LinearRegression()
reg.fit(X_train, y_train)

print(reg.score(X_test, y_test))

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr().abs(), cmap='coolwarm', annot=True)
plt.show()
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
import statsmodels.api as sm

# In[] Loading the data & data preparation
data = pd.read_csv('xAPI-Edu-Data.csv')
dependent_variable = 'gender_F'

transformedData = pd.get_dummies(data)
X = transformedData.drop(dependent_variable, axis=1)
X = X.drop('gender_M', axis=1)
y_true = transformedData[dependent_variable]

# In[] Linear regression
X_OLS = sm.add_constant(X)
model = sm.OLS(y_true,X_OLS)
results = model.fit()
y_pred = results.predict(X_OLS)

MAE = mean_absolute_error(y_true,y_pred)
MSE = mean_squared_error(y_true,y_pred)
print('MAE & MSE linear regression: ', (MAE, MSE))

# In[] Logistic regression
results = LogisticRegression().fit(X, y_true)
y_pred = results.predict(X)

MAE = mean_absolute_error(y_true,y_pred)
MSE = mean_squared_error(y_true,y_pred)
print('MAE & MSE logistic regression: ', (MAE, MSE))

# In[] Lasso regression
model = Lasso(alpha=1.0)
results = model.fit(X,y_true)
y_pred = results.predict(X)

MAE = mean_absolute_error(y_true,y_pred)
MSE = mean_squared_error(y_true,y_pred)
print('MAE & MSE lasso regression: ', (MAE, MSE))

# In[] Ridge regression
model = Ridge(alpha=1.0)
results = model.fit(X,y_true)
y_pred = results.predict(X)

MAE = mean_absolute_error(y_true,y_pred)
MSE = mean_squared_error(y_true,y_pred)
print('MAE & MSE ridge regression: ', (MAE, MSE))

# In[] Bayesian linear regression
model = BayesianRidge(compute_score = True)
results = model.fit(X,y_true)
y_pred = results.predict(X)

MAE = mean_absolute_error(y_true,y_pred)
MSE = mean_squared_error(y_true,y_pred)
print('MAE & MSE bayesian linear regression: ', (MAE, MSE))



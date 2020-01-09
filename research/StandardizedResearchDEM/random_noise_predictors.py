import numpy as np
from scipy.stats import pearsonr
from ninolearn.IO.read_processed import data_reader
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
reader = data_reader(startdate='1960-01', enddate='2017-12')

# ONI index is target variable
y = reader.read_csv('oni')

X = np.random.uniform(size=(len(y), 1000))

model = LinearRegression()

# =============================================================================
# Fit and evaluate on entire data set
# =============================================================================
model.fit(X, y)
oni_predict = model.predict(X)

plt.figure()
plt.title('Perfect fit when no data split is applied')
plt.plot(y.values)
plt.plot(oni_predict)

r,p = pearsonr(oni_predict, y)
print(f'r:{r}, p: {p}' )

# =============================================================================
# Fit on train data set and evaluate on test data set
# =============================================================================
Xtrain, Xtest = X[:500], X[500:]
ytrain, ytest = y[:500], y[500:]

model.fit(Xtrain, ytrain)
y_predict = model.predict(Xtest)

plt.figure()
plt.title('No skill becomes apparent')

plt.plot(ytest.values)
plt.plot(y_predict)

r,p = pearsonr(y_predict, ytest)
print(f'r:{r}, p: {p}' )
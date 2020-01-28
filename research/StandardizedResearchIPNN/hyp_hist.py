import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D
regressor = RandomForestRegressor(n_estimators=5)

plt.close("all")
df = pd.read_csv('/home/paul/Dokumente/data/ninolearn/model/dem_review_simple_decade1_lead0/hyperparameters_history.csv', index_col=0)
#df = pd.read_csv('/tmp/hyperparameter_history.csv', index_col=0)
y = df.loss
features = list(df.columns[1:])
X = df[features]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

regressor = regressor.fit(Xtrain, ytrain)
ypred =  regressor.predict(Xtest)

r, p = pearsonr(ypred, ytest)
r2 = r**2

fig = plt.figure()

plt.scatter(df.l1_hidden, df.loss)



#h = df.l1_hidden[df.loss>df.loss.quantile(q=0.5)]
#l = df.l1_hidden[df.loss<=df.loss.quantile(q=0.5)]
#
#l_mu, l_sigma = norm.fit(np.log10(l))
#h_mu, h_sigma = norm.fit(np.log10(h))
#
#x_sample = np.linspace(-16, 1000, 1000)
#
#l_sample = norm.pdf(x_sample, l_mu, l_sigma)
#h_sample = norm.pdf(x_sample, h_mu, h_sigma)
#
#ratio = l_sample/h_sample
#
#
#
#selcted = x_sample[ratio>np.quantile(ratio, 0.5)]

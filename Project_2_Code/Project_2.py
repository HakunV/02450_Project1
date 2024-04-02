from ucimlrepo import fetch_ucirepo
import importlib_resources
import numpy as np
from dtuimldmtools import similarity
from scipy.linalg import svd
from scipy.stats import zscore
import pandas as pd
import sklearn.linear_model as lm

from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel, axis, grid, bar, xticks, table, hist, ylim, subplot, boxplot, scatter

filename = importlib_resources.files("dtuimldmtools").joinpath("data/glass.data")

data = pd.read_csv(filename, sep=",", header=None, index_col=0)

# Shifting the classnumber of classes after 4 down,
# because there are no observations for class 4
data[10] = data[10].replace({5: 4})
data[10] = data[10].replace({6: 5})
data[10] = data[10].replace({7: 6})

arrData = np.array(data.values, dtype=np.float64)

X = arrData[:, :-1].copy()
y = arrData[:, -1].copy()

attributeNames = ["Ri", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

# Class 4 is also removed from the classNames, so there are only 6 classes
classNames = [
    "building_windows_float_processed", "building_windows_non_float_processed",
    "vehicle_windows_float_processed", "containers", "tableware", "headlamps"
]

N = len(y)
M = len(attributeNames)
C = len(classNames)

# Regression Part a: Linear Regression, trying to predict the amount of sodium in the glass

X_stand = X - np.ones((N, 1)) * X.mean(axis=0)
X_stand = X_stand * (1 / np.std(X_stand, 0))

refractive_idx = attributeNames.index("Ri")
y_reg = X_stand[:, refractive_idx]

X_cols = list(range(0, refractive_idx)) + list(range(refractive_idx+1, len(attributeNames)))
X_stand = X_stand[:, X_cols]

model = lm.LinearRegression()
model.fit(X_stand, y_reg)

y_est = model.predict(X_stand)
residual = y_est - y_reg

figure()
subplot(2, 1, 1)
plot(y_reg, y_est, ".")
xlabel("Refractive index (true)")
ylabel("Refractive index (estimated)")
subplot(2, 1, 2)
hist(residual, 40)

show()
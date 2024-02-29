from ucimlrepo import fetch_ucirepo
import importlib_resources
import numpy as np
from dtuimldmtools import similarity
from scipy.linalg import svd
from scipy.stats import zscore
import pandas as pd

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

# As can be seen in sums, all the attributes sums approx. to 100 in each observation
sums = [X[range(0, 214), 1] + X[range(0, 214), 2] + X[range(0, 214), 3] + X[range(0, 214), 4] + 
        X[range(0, 214), 5] + X[range(0, 214), 6] + X[range(0, 214), 7] + X[range(0, 214), 8]]

print(sums)

# Creating a matrix with basic statistics for each attribute
stats = np.zeros((4, M))
for a in range(M):
    stats[0, a] = np.round(X[:, a].mean(), decimals=3)
    stats[1, a] = np.round(np.median(X[:, a]), decimals=3)
    stats[2, a] = np.round(X[:, a].var(), decimals=3)
    stats[3, a] = np.round(X[:, a].std(), decimals=3)

# Visualizing the statistics in a table
f1 = figure(1)

f1.patch.set_visible(False)
axis('off')
axis('tight')

t1 = table(cellText=stats, colLabels=attributeNames, rowLabels=["Mean", "Median", "Variance", "STD"], loc='center')

t1.scale(1, 2)
t1.set_fontsize(16)

f1.tight_layout()


# Calculating the similarities of all the attributes
sim = similarity(X[:, range(0, M)].T, X[:, range(0, M)].T, 'Correlation').round(decimals=5)

# Changing the reoccuring coefficients to 0
# under the diagonal line with ones
for k in range(0, M):
    sim[range(k+1, M), k] = 0


# Visualizing the similarities in a table
f2 = figure(2)
title("Correlation")

f2.patch.set_visible(False)
axis('off')
axis('tight')

t2 = table(cellText=sim, rowLabels=attributeNames, colLabels=attributeNames, loc='center')

t2.scale(1, 2)
t2.set_fontsize(32)

f2.tight_layout()


# Ri and Ca was very correlated, so we plot them together here
# to visualize it better
f3 = figure(3)

plot(X[:, 0], X[:, 6], "o")


# Generating histograms of all the attributes, to see how they are distributed
f4 = figure(4, figsize=(10, 10))

u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)

for i in range(M):
    subplot(int(u), int(v), i+1)
    hist(X[:, i], 15)
    xlabel(attributeNames[i])
    ylim(0, N/2)
    

# Generating boxplots to spot outliers
f5 = figure(5)
title("Boxplot of attributes")

boxplot(zscore(X, ddof=1), attributeNames)
xticks(range(1, 10), attributeNames, rotation=45)


# Standardizing the data
Y = X - np.ones((N, 1)) * X.mean(axis=0)
Y = Y * (1 / np.std(Y, 0))

# Doing svd to calculate variance
U, S, Vh = svd(Y, full_matrices=False)

# Individual variance
rho = (S*S) / (S*S).sum()

threshold = 0.9

# Plotting the individual and cumulative variances in a plot
f7 = figure(7)
title("PCA")
plot(range(1, len(rho) + 1), rho, "x-")
plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plot([1, len(rho)], [threshold, threshold], "k--")
xlabel("Principal Component")
ylabel("Variance Explained")
legend(["Individual", "Cumulative", "Threshold"])
grid()


# Transposing, so the vectors are in columns
V = Vh.T

print(V[:, :4])

# A table to show the coordinates of the first 4 PC's
f10 = figure(10)

f10.patch.set_visible(False)
axis('off')
axis('tight')

t10 = table(cellText=np.round(V[:, :4], decimals=5), colLabels=["PC1", "PC2", "PC3", "PC4"], loc='center')

t10.scale(1, 2)
t10.set_fontsize(20)

f10.tight_layout()


# Projecting the data on the first 3 PC's
Z = Y @ V[:, :3]
colors = ["blue", "green", "red", "yellow", "brown", "gray"]

# Displaying the projected data in 3D
f8 = figure(8)
title("Data projected on the first 3 PC's")
ax = f8.add_subplot(111, projection="3d")

for c in range(C):
    class_mask = y == c
    ax.scatter(Z[class_mask, 0], Z[class_mask, 1], Z[class_mask, 2], c=colors[c])

ax.view_init(30, 220)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# A barograph to show each components affect on each attribute
pcs = [0, 1, 2, 3]
legendStr = ["PC" + str(e + 1) for e in pcs]
bw = 0.2
r = np.arange(1, M+1)

f9 = figure(9)
title("PCA Component Coefficient")
for i in pcs:
    bar(r + i * bw, V[:, i], width=bw)
xticks(r + bw, attributeNames)
xlabel("Attributes")
ylabel("Component Coefficient")
legend(legendStr)
grid()
show()
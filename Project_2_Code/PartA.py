from ucimlrepo import fetch_ucirepo
import importlib_resources
import numpy as np
from dtuimldmtools import similarity
from scipy.linalg import svd
from scipy.stats import zscore
import pandas as pd
import sklearn.linear_model as lm
import torch
from tabulate import tabulate
import scipy.stats as st

from sklearn import model_selection
from dtuimldmtools import (
    rlr_validate,
    draw_neural_net,
    train_neural_net,  
)

from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel, axis, grid, bar, xticks, table, hist, ylim, subplot, subplots, boxplot, scatter, loglog, semilogx

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

# Make a list for storing assigned color of learning curve for up to K=10
color_list = [
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:red",
    "tab:blue",
]

# Linear Regression, trying to predict the refractive index of the glass

refractive_idx = attributeNames.index("Ri")
y_reg = X[:, refractive_idx]

X_cols = list(range(0, refractive_idx)) + list(range(refractive_idx+1, len(attributeNames)))
X = X[:, X_cols]

attributeNames = attributeNames[1:len(X_cols)+1]
M = M - 1

# Add offset attribute
X_off = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M + 1

# Crossvalidation
# Create crossvalidation partition for evaluation
K = 2
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.0, range(-5, 9))

# Initialize variables
# T = len(lambdas)
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV.split(X_off, y_reg):
    # extract training and test set for current CV fold
    X_train = X_off[train_index]
    y_train = y_reg[train_index]
    X_test = X_off[test_index]
    y_test = y_reg[test_index]
    internal_cross_validation = 10

    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = (
        np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test[k] = (
        np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    )

    # Display the results for the last cross-validation fold
    if k == K - 1:
        figure(k, figsize=(12, 8))
        subplot(1, 2, 1)
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
        xlabel("Regularization factor")
        ylabel("Mean Coefficient Values")
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        legend(attributeNames[1:], loc='best')

        subplot(1, 2, 2)
        title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
        loglog(
            lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
        )
        xlabel("Regularization factor")
        ylabel("Squared error (crossvalidation)")
        legend(["Train error", "Validation error"])
        grid()
        
        y_est_rlr = X_test @ w_rlr[:, k]

        figure()
        title("Best Weights Predictions")
        plot(y_test, y_est_rlr, ".g")
        xlabel("Refractive index (true)")
        ylabel("Refractive index (estimated)")
        
        print("Best Weights:")
        for m in range(M):
            print("{:>15} {:>15}".format(attributeNames[m], np.round(w_rlr[m, k], 5)))

    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1


print("Optimal Error RLR")
print(Error_test_rlr)

show()
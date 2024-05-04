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

from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel, axis, grid, bar, xticks, table, hist, ylim, subplot, subplots, boxplot, scatter, loglog, semilogx, errorbar

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
K = 6
outer_CV = model_selection.KFold(K, shuffle=True)
# CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.0, range(-5, 9))

summaries, summaries_axes = subplots(1, 2, figsize=(10, 5))

# Parameters for neural network classifier
hidden_units = range(1, 11)  # number of hidden units
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 10000

# Initialize variables
# T = len(lambdas)
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
w_noreg = np.empty((M, K))

outer_mu_rlr = np.empty((K, M - 1))
outer_sigma_rlr = np.empty((K, M - 1))
outer_mu_ann = np.empty((K, M - 1))
outer_sigma_ann = np.empty((K, M - 1))
inner_mu = np.empty((K, M - 1))
inner_sigma = np.empty((K, M - 1))

opt_lambdas = np.empty((K, 1))
opt_hiddens = np.empty((K, 1))

errors_ann = np.empty((K, 1))  # make a list for storing generalizaition error for ann in each loop

CI_LandA_arr = np.empty((K, 2))
CI_LandB_arr = np.empty((K, 2))
CI_AandB_arr = np.empty((K, 2))

p_LandA_arr = np.empty((K, 1))
p_LandB_arr = np.empty((K, 1))
p_AandB_arr = np.empty((K, 1))

k = 0
for outer_train_index, outer_test_index in outer_CV.split(X_off, y_reg):
    
    outer_X_train = X_off[outer_train_index]
    outer_y_train = y_reg[outer_train_index]
    outer_X_test = X_off[outer_test_index]
    outer_y_test = y_reg[outer_test_index]
    
    inner_CV = model_selection.KFold(K, shuffle=True)
    
    error_val_rlr = 1000
    opt_lambda_rlr = 0
    
    min_final_loss = 1000
    opt_h = 0
    
    i = 0
    for inner_train_index, inner_test_index in inner_CV.split(outer_X_train, outer_y_train):
        
        inner_X_train = outer_X_train[inner_train_index]
        inner_y_train = outer_y_train[inner_train_index]
        
        (
            opt_val_err,
            opt_lambda,
            mean_w_vs_lambda,
            train_err_vs_lambda,
            test_err_vs_lambda,
        ) = rlr_validate(inner_X_train, inner_y_train, lambdas)
        
        if opt_val_err < error_val_rlr:
            error_val_rlr = opt_val_err
            opt_lambda_rlr = opt_lambda
        
        inner_mu[i, :] = np.mean(outer_X_train[:, 1:], 0)
        inner_sigma[i, :] = np.std(outer_X_train[:, 1:], 0)

        outer_X_train_stand = (outer_X_train[:, 1:] - inner_mu[i, :]) / inner_sigma[i, :]
        outer_X_train_stand = np.concatenate((np.ones((outer_X_train_stand.shape[0], 1)), outer_X_train_stand), 1)
            
        inner_X_train_ten = torch.Tensor(outer_X_train_stand[inner_train_index, :])
        inner_y_train_ten = torch.Tensor(outer_y_train[inner_train_index]).unsqueeze(1)
            
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, hidden_units[i]),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(hidden_units[i], 1),  # n_hidden_units to 1 output neuron
            # no final transfer function, i.e. "linear output"
        )
        loss_fn = torch.nn.MSELoss()  # This is now a mean-squared-error loss
        
        net, final_loss, learning_curve = train_neural_net(
            model,
            loss_fn,
            X=inner_X_train_ten,
            y=inner_y_train_ten,
            n_replicates=n_replicates,
            max_iter=max_iter,
        )
        
        if final_loss < min_final_loss:
            min_final_loss = final_loss
            opt_h = hidden_units[i]
        
        i += 1
        
    opt_lambdas[k] = opt_lambda_rlr
    opt_hiddens[k] = opt_h
    
    # Standardize
    outer_mu_rlr[k, :] = np.mean(outer_X_train[:, 1:], 0)
    outer_sigma_rlr[k, :] = np.std(outer_X_train[:, 1:], 0)
    
    outer_X_train[:, 1:] = (outer_X_train[:, 1:] - outer_mu_rlr[k, :]) / outer_sigma_rlr[k, :]
    outer_X_test[:, 1:] = (outer_X_test[:, 1:] - outer_mu_rlr[k, :]) / outer_sigma_rlr[k, :]
    
    Xty = outer_X_train.T @ outer_y_train
    XtX = outer_X_train.T @ outer_X_train
    
    outer_mu_ann[k, :] = np.mean(X_off[:, 1:], 0)
    outer_sigma_ann[k, :] = np.std(X_off[:, 1:], 0)

    X_stand = (X_off[:, 1:] - outer_mu_ann[k, :]) / outer_sigma_ann[k, :]
    X_stand = np.concatenate((np.ones((X_stand.shape[0], 1)), X_stand), 1)
    
    outer_X_train_ten = torch.Tensor(X_stand[outer_train_index, :])
    outer_y_train_ten = torch.Tensor(y_reg[outer_train_index]).unsqueeze(1)
    outer_X_test_ten = torch.Tensor(X_stand[outer_test_index, :])
    outer_y_test_ten = torch.Tensor(y_reg[outer_test_index]).unsqueeze(1)
    
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, opt_h),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(opt_h, 1),  # n_hidden_units to 1 output neuron
        # no final transfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=outer_X_train_ten,
        y=outer_y_train_ten,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(outer_y_train - outer_y_train.mean()).sum(axis=0) / outer_y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(outer_y_test - outer_y_train.mean()).sum(axis=0) / outer_y_test.shape[0]
    )

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda_rlr * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = (
        np.square(outer_y_train - outer_X_train @ w_rlr[:, k]).sum(axis=0) / outer_y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(outer_y_test - outer_X_test @ w_rlr[:, k]).sum(axis=0) / outer_y_test.shape[0]
    )
    
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
        
        print("Best Weights:")
        for m in range(M):
            print("{:>15} {:>15}".format(attributeNames[m], np.round(w_rlr[m, -1], 5)))
        
        y_est_rlr = outer_X_test @ w_rlr[:, k]

        figure()
        title("Best Weights Predictions")
        plot(outer_y_test, y_est_rlr, ".g")
        xlabel("Refractive index (true)")
        ylabel("Refractive index (estimated)")

    y_test_est = net(outer_X_test_ten)

    # Determine errors
    se = (y_test_est.float() - outer_y_test_ten.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(outer_y_test_ten)).data.numpy() # mean
    errors_ann[k] = mse  # store error rate for current CV fold

    # Display the learning curve for the best net in the current fold
    (h,) = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label("CV fold {0}".format(k + 1))
    summaries_axes[0].set_xlabel("Iterations")
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel("Loss")
    summaries_axes[0].set_title("Learning curves")
    
    # Statistical evaluation
    z_rlr = np.abs(outer_y_test - outer_X_test @ w_rlr[:, k]) ** 2
    
    z_ann = np.array(se.detach()).reshape(-1)
        
    z_base = np.abs(outer_y_test - outer_y_train.mean()) ** 2
    
    alpha = 0.05

    print("Length:")
    print(len(z_ann))

    z_LandA = z_rlr - z_ann
    lower, upper = st.t.interval(
        1 - alpha, len(z_LandA) - 1, loc=np.mean(z_LandA), scale=st.sem(z_LandA)
    )
    CI_LandA_arr[k, 0] = lower
    CI_LandA_arr[k, 1] = upper
    
    p_LandA_arr[k] = 2 * st.t.cdf(-np.abs(np.mean(z_LandA)) / st.sem(z_LandA), df=len(z_LandA) - 1)

    
    z_LandB = z_rlr - z_base
    lower, upper = st.t.interval(
        1 - alpha, len(z_LandB) - 1, loc=np.mean(z_LandB), scale=st.sem(z_LandB)
    )
    CI_LandB_arr[k, 0] = lower
    CI_LandB_arr[k, 1] = upper
    
    p_LandB_arr[k] = 2 * st.t.cdf(-np.abs(np.mean(z_LandB)) / st.sem(z_LandB), df=len(z_LandB) - 1)
    
    
    z_AandB = z_ann - z_base
    lower, upper = st.t.interval(
        1 - alpha, len(z_AandB) - 1, loc=np.mean(z_AandB), scale=st.sem(z_AandB)
    )
    CI_AandB_arr[k, 0] = lower
    CI_AandB_arr[k, 1] = upper
    
    p_AandB_arr[k] = 2 * st.t.cdf(-np.abs(np.mean(z_AandB)) / st.sem(z_AandB), df=len(z_AandB) - 1)
    
    
    print('Cross validation fold {0}/{1}:'.format(k+1,K))

    k += 1

# Number of intervals
n_intervals = CI_LandA_arr.shape[0]

# Create an array of x values (e.g., [0, 1] to represent each interval)
x_values = np.arange(n_intervals)

# Extract lower and upper bounds from the confidence intervals
lower_bounds_LandA = CI_LandA_arr[:, 0]
upper_bounds_LandA = CI_LandA_arr[:, 1]

lower_bounds_LandB = CI_LandB_arr[:, 0]
upper_bounds_LandB = CI_LandB_arr[:, 1]

lower_bounds_AandB = CI_AandB_arr[:, 0]
upper_bounds_AandB = CI_AandB_arr[:, 1]

# Plot the confidence intervals
figure()
subplot(1, 3, 1)
title("RLR and ANN")
errorbar(x_values, (lower_bounds_LandA + upper_bounds_LandA) / 2, yerr=[(upper_bounds_LandA - lower_bounds_LandA) / 2], fmt='o', capsize=5)
xticks(x_values, ['Interval 1', 'Interval 2', 'Interval 3', 'Interval 4', 'Interval 5', 'Interval 6'])
xlabel('Intervals')
ylabel('Generalization Error')

subplot(1, 3, 2)
title("RLR and Baseline")
errorbar(x_values, (lower_bounds_LandB + upper_bounds_LandB) / 2, yerr=[(upper_bounds_LandB - lower_bounds_LandB) / 2], fmt='o', capsize=5)
xticks(x_values, ['Interval 1', 'Interval 2', 'Interval 3', 'Interval 4', 'Interval 5', 'Interval 6'])
xlabel('Intervals')
ylabel('Generalization Error')

subplot(1, 3, 3)
title("ANN and Baseline")
errorbar(x_values, (lower_bounds_AandB + upper_bounds_AandB) / 2, yerr=[(upper_bounds_AandB - lower_bounds_AandB) / 2], fmt='o', capsize=5)
xticks(x_values, ['Interval 1', 'Interval 2', 'Interval 3', 'Interval 4', 'Interval 5', 'Interval 6'])
xlabel('Intervals')
ylabel('Generalization Error')



print("Optimal Hidden Units")
print(opt_hiddens)    

print("Optimal Lambdas")
print(opt_lambdas)

print("Optimal Error RLR")
print(Error_test_rlr)

print("Optimal Error Baseline")
print(Error_test_nofeatures)

print("Optimal Error ANN")
print(errors_ann)

# Display results
print("Regularized linear regression:")
print("- Training error: {0}".format(Error_train_rlr.mean()))
print("- Test error:     {0}".format(Error_test_rlr.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train_rlr.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test_rlr.sum())
        / Error_test_nofeatures.sum()
    )
)

print("Confidence intervals between RLR and ANN")
print(CI_LandA_arr)

print("p-values between RLR and ANN")
print(p_LandA_arr)

print("Confidence intervals between RLR and Baseline")
print(CI_LandB_arr)

print("p-values between RLR and Baseline")
print(p_LandB_arr)

print("Confidence intervals between ANN and Baseline")
print(CI_AandB_arr)

print("p-values between ANN and Baseline")
print(p_AandB_arr)

show()
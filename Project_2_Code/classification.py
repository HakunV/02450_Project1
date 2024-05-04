import numpy as np
import pandas as pd
import importlib_resources
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from scipy import stats

# Adjusted loading of the dataset
filename = importlib_resources.files("dtuimldmtools").joinpath("data/glass.data")
data = pd.read_csv(filename, sep=",", header=None, index_col=0)

# Adjust class numbers due to missing class 4
data[10] = data[10].replace({5: 4, 6: 5, 7: 6})

# Preparing data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Set attribute and class names (for plotting)
attributeNames = ["Ri", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
classNames = ["building_windows_float_processed", "building_windows_non_float_processed",
              "vehicle_windows_float_processed", "containers", "tableware", "headlamps"]

# Convert class labels from original to {0,1,...,C-1}
classLabels = np.unique(y)
classDict = dict(zip(classLabels, range(len(classLabels))))
y = np.array([classDict[cl] for cl in y])

# Initialize cross-validation method
K = 10
CV = model_selection.StratifiedKFold(n_splits=K, shuffle=True)

# Initialize models
logistic_model = LogisticRegression(max_iter=10000)
dummy_model = DummyClassifier(strategy='most_frequent')
knn_model = KNeighborsClassifier(n_neighbors=5)  # Using K=5 for KNN

# Initialize variables for performance tracking
logistic_errors = np.zeros(K)
dummy_errors = np.zeros(K)
knn_errors = np.zeros(K)
fold_details = []

k = 0
for (train_index, test_index) in CV.split(X, y):
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]

    # Train models
    logistic_model.fit(X_train, y_train)
    dummy_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    # Predict and calculate error rates
    logistic_error = 1 - accuracy_score(y_test, logistic_model.predict(X_test))
    dummy_error = 1 - accuracy_score(y_test, dummy_model.predict(X_test))
    knn_error = 1 - accuracy_score(y_test, knn_model.predict(X_test))
    
    logistic_errors[k] = logistic_error
    dummy_errors[k] = dummy_error
    knn_errors[k] = knn_error

    # Store fold results
    fold_details.append((k+1, logistic_error, knn_error, dummy_error))

    k += 1

# Output average error rates
print(f'Logistic Regression error rate: {logistic_errors.mean()}')
print(f'Dummy (Baseline) error rate: {dummy_errors.mean()}')
print(f'KNN error rate: {knn_errors.mean()}')

# Statistical evaluation
# Paired t-test between Logistic Regression and KNN
t_stat, p_val = stats.ttest_rel(logistic_errors, knn_errors)
print(f'Paired t-test between Logistic Regression and KNN, p-value: {p_val}')

# Print detailed fold results
print("Fold | Logistic Regression Error | KNN Error | Baseline Error")
for detail in fold_details:
    print(f"{detail[0]} | {detail[1]:.3f} | {detail[2]:.3f} | {detail[3]:.3f}")

# Displaying results in a plot for visualization
plt.figure(figsize=(12, 6))
plt.boxplot([logistic_errors, dummy_errors, knn_errors], labels=['Logistic Regression', 'Dummy', 'KNN'])
plt.ylabel('Error rate')
plt.title('Model comparison on Glass Identification Dataset')
plt.show()

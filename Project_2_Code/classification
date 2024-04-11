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
CV = model_selection.StratifiedKFold(K, shuffle=True)

# Initialize models
logistic_model = LogisticRegression(max_iter=10000)
dummy_model = DummyClassifier(strategy='most_frequent')
knn_model = KNeighborsClassifier(n_neighbors=5)  # Using K=5 for KNN

# Initialize variables for performance tracking
logistic_errors = np.zeros(K)
dummy_errors = np.zeros(K)
knn_errors = np.zeros(K)

k = 0
for (train_index, test_index) in CV.split(X, y):
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]

    # Train models
    logistic_model.fit(X_train, y_train)
    dummy_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    # Predict and calculate error rates
    logistic_errors[k] = 1 - accuracy_score(y_test, logistic_model.predict(X_test))
    dummy_errors[k] = 1 - accuracy_score(y_test, dummy_model.predict(X_test))
    knn_errors[k] = 1 - accuracy_score(y_test, knn_model.predict(X_test))

    k += 1

# Output average error rates
print(f'Logistic Regression error rate: {logistic_errors.mean()}')
print(f'Dummy (Baseline) error rate: {dummy_errors.mean()}')
print(f'KNN error rate: {knn_errors.mean()}')

# Statistical evaluation
# Paired t-test between Logistic Regression and KNN
t_stat, p_val = stats.ttest_rel(logistic_errors, knn_errors)
print(f'Paired t-test between Logistic Regression and KNN, p-value: {p_val}')

# Displaying results in a plot for visualization
plt.figure(figsize=(12, 6))
plt.boxplot([logistic_errors, dummy_errors, knn_errors])
plt.xticks([1, 2, 3], ['Logistic Regression', 'Dummy', 'KNN'])
plt.ylabel('Error rate')
plt.title('Model comparison on Glass Identification Dataset')
plt.show()

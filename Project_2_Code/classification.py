import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import importlib_resources

# Load the dataset
filename = importlib_resources.files("dtuimldmtools").joinpath("data/glass.data")
data = pd.read_csv(filename, sep=",", header=None, index_col=0)

# Adjusting class labels similar to the regression part
data[10] = data[10].replace({5: 4, 6: 5, 7: 6})

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize features
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X)  # Standardized dataset

# Logistic Regression with Cross-Validation for regularization strength
log_reg_cv = LogisticRegressionCV(cv=5, max_iter=10000, random_state=42).fit(X_std, y)

# Finding the best k for k-Nearest Neighbors using cross-validation
k_values = range(1, 11)
best_score = 0
best_k = 1
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_std, y, cv=kf, scoring='accuracy')
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_k = k
knn_best = KNeighborsClassifier(n_neighbors=best_k)

# Perform cross-validation for statistical comparison between Logistic Regression CV and best KNN
log_reg_scores = cross_val_score(log_reg_cv, X_std, y, cv=kf, scoring='accuracy')
knn_best_scores = cross_val_score(knn_best, X_std, y, cv=kf, scoring='accuracy')

# Statistical test
t_stat, p_val = ttest_rel(log_reg_scores, knn_best_scores)
print(f'Paired t-test between Logistic Regression CV and best KNN (k={best_k}) p-value: {p_val:.4f}')

# Visualization
plt.figure(figsize=(10, 6))
plt.boxplot([log_reg_scores, knn_best_scores], labels=['Logistic Regression CV', f'Best KNN (k={best_k})'])
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()

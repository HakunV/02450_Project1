import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
data = pd.read_csv('C:/Users/sugul/anaconda3/envs/machine_learning/Lib/site-packages/dtuimldmtools/data/glass.data', names=column_names, index_col='Id')

# Basic Statistical Analysis
stats_summary = data.describe().transpose()[['mean', 'std', '50%']]  # Including median (50%)

# Standardize the features before PCA
features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
x = data.loc[:, features].values
x = StandardScaler().fit_transform(x)

# PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# Adding 'Type' back for color-coding in visualization
finalDf = pd.concat([principalDf, data[['Type']].reset_index(drop=True)], axis = 1)

# Plotting
plt.figure(figsize=(10, 6))

# Distributions of a couple of features (RI and Na)
plt.subplot(1, 3, 1)
plt.hist(data['RI'], bins=20, alpha=0.7, label='RI')
plt.hist(data['Na'], bins=20, alpha=0.7, label='Na')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# PCA result plot
plt.subplot(1, 3, 2)
targets = np.unique(data['Type'])
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Type'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(targets)
plt.title('PCA of Glass Dataset')

# Show plots
plt.tight_layout()
plt.show()

stats_summary

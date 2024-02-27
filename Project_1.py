from ucimlrepo import fetch_ucirepo
import importlib_resources
import numpy as np
from dtuimldmtools import similarity

from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel, axis, scatter

filename = importlib_resources.files("dtuimldmtools").joinpath("data/glass.data")

with open(filename, "r") as f:
    raw_file = f.read()

delimiters = ["\n", ","]

for d in delimiters:
    raw_file = " ".join(raw_file.split(d))
    
corpus = raw_file.split()

corpus = list(filter(None, corpus))

startMat = np.asmatrix(corpus)

print("Document-term matrix analysis")
print()
print("Corpus (5 documents/sentences):")
print(startMat)
print()

refractive_index = np.array(startMat[0, range(1, startMat.size, 11)])
natrium = np.array(startMat[0, range(2, startMat.size, 11)])
magnesium = np.array(startMat[0, range(3, startMat.size, 11)])
aluminium = np.array(startMat[0, range(4, startMat.size, 11)])
silicon = np.array(startMat[0, range(5, startMat.size, 11)])
potassium = np.array(startMat[0, range(6, startMat.size, 11)])
calcium = np.array(startMat[0, range(7, startMat.size, 11)])
barium = np.array(startMat[0, range(8, startMat.size, 11)])
iron = np.array(startMat[0, range(2, startMat.size, 11)])

refractive_index_mean = refractive_index.mean()
refractive_index_var = refractive_index.var()
refractive_index_std = refractive_index.std(ddof=1)

natrium_mean = natrium.mean()
natrium_var = natrium.var()
natrium_std = natrium.std(ddof=1)

magnesium_mean = magnesium.mean()
magnesium_var = magnesium.var()
magnesium_std = magnesium.std(ddof=1)

aluminium_mean = aluminium.mean()
aluminium = aluminium.var()
magnesium_std = aluminium.std(ddof=1)

figure()
title("Refractive index")

scatter(refractive_index[0, :], refractive_index[0, :])

axis("tight")

show()

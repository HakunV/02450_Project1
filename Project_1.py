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

refractive_index = np.array(np.array(startMat[0, range(1, startMat.size, 11)])[0])
refractive_index = np.array([float(i) for i in refractive_index])
natrium = np.array(np.array(startMat[0, range(2, startMat.size, 11)])[0])
natrium = np.array([float(i) for i in natrium])
magnesium = np.array(np.array(startMat[0, range(3, startMat.size, 11)])[0])
magnesium = np.array([float(i) for i in magnesium])
aluminium = np.array(np.array(startMat[0, range(4, startMat.size, 11)])[0])
aluminium = np.array([float(i) for i in aluminium])
silicon = np.array(np.array(startMat[0, range(5, startMat.size, 11)])[0])
silicon = np.array([float(i) for i in silicon])
potassium = np.array(np.array(startMat[0, range(6, startMat.size, 11)])[0])
potassium = np.array([float(i) for i in potassium])
calcium = np.array(np.array(startMat[0, range(7, startMat.size, 11)])[0])
calcium = np.array([float(i) for i in calcium])
barium = np.array(np.array(startMat[0, range(8, startMat.size, 11)])[0])
barium = np.array([float(i) for i in barium])
iron = np.array(np.array(startMat[0, range(2, startMat.size, 11)])[0])
iron = np.array([float(i) for i in iron])

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
aluminium_var = aluminium.var()
aluminium_std = aluminium.std(ddof=1)

silicon_mean = silicon.mean()
silicon_var = silicon.var()
silicon_std = silicon.std(ddof=1)

potassium_mean = potassium.mean()
potassium_var = potassium.var()
potassium_std = potassium.std(ddof=1)

calcium_mean = calcium.mean()
calcium_var = calcium.var()
calcium_std = calcium.std(ddof=1)

barium_mean = barium.mean()
barium_var = barium.var()
barium_std = barium.std(ddof=1)

iron_mean = iron.mean()
iron_var = iron.var()
iron_std = iron.std(ddof=1)

# sim = similarity(refractive_index, aluminium, 'Cosine')

# figure()
# title("Refractive index")

# scatter(refractive_index[0, :], refractive_index[0, :])

# axis("tight")

# show()

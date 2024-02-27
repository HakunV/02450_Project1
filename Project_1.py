from ucimlrepo import fetch_ucirepo
import importlib_resources
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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

natrium = np.array(startMat[0, range(2, startMat.size, 11)])

print(natrium)
print(len(natrium[0]))

# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X = glass_identification.data.features 
y = glass_identification.data.targets 
  

# # metadata 
# print(glass_identification.metadata) 
  
# # variable information
# print(glass_identification.variables) 

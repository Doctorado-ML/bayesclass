from bayesclass.clfs import AODENew, TANNew, KDBNew, AODE
from benchmark.datasets import Datasets
import os

os.chdir("../discretizbench")
dt = Datasets()
clfan = AODENew()
clftn = TANNew()
clfkn = KDBNew()
# clfa = AODE()
X, y = dt.load("iris")
# clfa.fit(X, y)
clfan.fit(X, y)
clftn.fit(X, y)
clfkn.fit(X, y)


self.discretizer_.target_
self.estimator.indexed_features_

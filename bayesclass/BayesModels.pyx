# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
import numpy as np

cdef extern from "cpp/TAN.h" namespace "bayesnet":
    cdef cppclass TAN:
        TAN() except + 
        void fit(vector[vector[int]]&, vector[int]&, vector[string]&, string, map[string, vector[int]]&)
        vector[int] predict(vector[vector[int]]&)
        vector[vector[double]] predict_proba(vector[vector[int]]&)
        float score(const vector[vector[int]]&, const vector[int]&)
        vector[string] graph()
        
cdef class CTAN:
    cdef TAN *thisptr
    def __cinit__(self):
        self.thisptr = new TAN() 
    def __dealloc__(self):
        del self.thisptr
    def fit(self, X, y, features, className, states):
        X_ = [X[:, i] for i in range(X.shape[1])]
        features_bytes = [x.encode() for x in features]
        self.thisptr.fit(X_, y, features_bytes, className.encode(), states)
        return self
    def predict(self, X):
        X_ = [X[:, i] for i in range(X.shape[1])]
        return self.thisptr.predict(X_)
    def score(self, X, y):
        X_ = [X[:, i] for i in range(X.shape[1])]
        return self.thisptr.score(X_, y)
    def graph(self):
        return self.thisptr.graph()
    def __reduce__(self):
        return (CTAN, ())

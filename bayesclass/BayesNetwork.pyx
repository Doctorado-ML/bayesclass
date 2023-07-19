# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
import numpy as np

cdef extern from "cpp/Network.h" namespace "bayesnet":
    cdef cppclass Network:
        Network(float, float) except + 
        void fit(vector[vector[int]]&, vector[int]&, vector[string]&, string)
        vector[int] predict(vector[vector[int]]&)
        vector[vector[double]] predict_proba(vector[vector[int]]&)
        float score(const vector[vector[int]]&, const vector[int]&)
        void addNode(string, int)
        void addEdge(string, string) except +
        vector[string] getFeatures()
        int getClassNumStates()
        int getStates()
        string getClassName()
        string version()
        void show()
        
cdef class BayesNetwork:
    cdef Network *thisptr
    def __cinit__(self, maxThreads=0.8, laplaceSmooth=1.0):
        self.thisptr = new Network(maxThreads, laplaceSmooth) 
    def __dealloc__(self):
        del self.thisptr
    def fit(self, X, y, features, className):
        X_ = [X[:, i] for i in range(X.shape[1])]
        features_bytes = [x.encode() for x in features]
        self.thisptr.fit(X_, y, features_bytes, className.encode())
        return self
    def predict(self, X):
        X_ = [X[:, i] for i in range(X.shape[1])]
        return self.thisptr.predict(X_)
    def predict_proba(self, X):
        X_ = [X[:, i] for i in range(X.shape[1])]
        return self.thisptr.predict_proba(X_)
    def score(self, X, y):
        X_ = [X[:, i] for i in range(X.shape[1])]
        return self.thisptr.score(X_, y)
    def addNode(self, name, states):
        self.thisptr.addNode(str.encode(name), states)
    def addEdge(self, source, destination):
        self.thisptr.addEdge(str.encode(source), str.encode(destination))
    def getFeatures(self):
        res = self.thisptr.getFeatures()
        return [x.decode() for x in res]
    def getStates(self):
        return self.thisptr.getStates()
    def getClassName(self):
        return self.thisptr.getClassName().decode()
    def getClassNumStates(self):
        return self.thisptr.getClassNumStates()
    def show(self):
        return self.thisptr.show()
    def __reduce__(self):
        return (BayesNetwork, ())

cdef extern from "cpp/Metrics.hpp" namespace "bayesnet":
    cdef cppclass Metrics:
        Metrics(vector[vector[int]], vector[int], vector[string]&, string&, int) except +
        vector[float] conditionalEdgeWeights()

cdef class CMetrics:
    cdef Metrics *thisptr
    def __cinit__(self, X, y, features, className, classStates):
        X_ = [X[:, i] for i in range(X.shape[1])]
        features_bytes = [x.encode() for x in features]
        self.thisptr = new Metrics(X_, y, features_bytes, className.encode(), classStates)
    def __dealloc__(self):
        del self.thisptr
    def conditionalEdgeWeights(self, n_vars):
        return np.reshape(self.thisptr.conditionalEdgeWeights(), (n_vars, n_vars))
    def __reduce__(self):
        return (CMetrics, ())

cdef extern from "cpp/TAN.h" namespace "bayesnet":
    cdef cppclass CTAN:
        CTAN() except + 
        void fit(vector[vector[int]]&, vector[int]&, vector[string]&, string, map[string, vector[int]]&)
        vector[int] predict(vector[vector[int]]&)
        vector[vector[double]] predict_proba(vector[vector[int]]&)
        float score(const vector[vector[int]]&, const vector[int]&)
        vector[string] graph()

cdef extern from "cpp/KDB.h" namespace "bayesnet":
    cdef cppclass CKDB:
        CKDB(int) except + 
        void fit(vector[vector[int]]&, vector[int]&, vector[string]&, string, map[string, vector[int]]&)
        vector[int] predict(vector[vector[int]]&)
        vector[vector[double]] predict_proba(vector[vector[int]]&)
        float score(const vector[vector[int]]&, const vector[int]&)
        vector[string] graph()

cdef extern from "cpp/AODE.h" namespace "bayesnet":
    cdef cppclass CAODE:
        CAODE() except + 
        void fit(vector[vector[int]]&, vector[int]&, vector[string]&, string, map[string, vector[int]]&)
        vector[int] predict(vector[vector[int]]&)
        vector[vector[double]] predict_proba(vector[vector[int]]&)
        float score(const vector[vector[int]]&, const vector[int]&)
        vector[string] graph()
        
cdef class TAN:
    cdef CTAN *thisptr
    def __cinit__(self):
        self.thisptr = new CTAN() 
    def __dealloc__(self):
        del self.thisptr
    def fit(self, X, y, features, className, states):
        X_ = [X[:, i] for i in range(X.shape[1])]
        features_bytes = [x.encode() for x in features]
        states_dict = {key.encode(): value for key, value in states.items()}
        states_dict[className.encode()] = np.unique(y).tolist()
        self.thisptr.fit(X_, y, features_bytes, className.encode(), states_dict)
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
        return (TAN, ())

cdef class CKDB:
    cdef KDB *thisptr
    def __cinit__(self, k):
        self.thisptr = new KDB(k) 
    def __dealloc__(self):
        del self.thisptr
    def fit(self, X, y, features, className, states):
        X_ = [X[:, i] for i in range(X.shape[1])]
        features_bytes = [x.encode() for x in features]
        states_dict = {key.encode(): value for key, value in states.items()}
        states_dict[className.encode()] = np.unique(y).tolist()
        self.thisptr.fit(X_, y, features_bytes, className.encode(), states_dict)
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
        return (CKDB, ())

cdef class CAODE:
    cdef AODE *thisptr
    def __cinit__(self):
        self.thisptr = new AODE() 
    def __dealloc__(self):
        del self.thisptr
    def fit(self, X, y, features, className, states):
        X_ = [X[:, i] for i in range(X.shape[1])]
        features_bytes = [x.encode() for x in features]
        states_dict = {key.encode(): value for key, value in states.items()}
        states_dict[className.encode()] = np.unique(y).tolist()
        self.thisptr.fit(X_, y, features_bytes, className.encode(), states_dict)
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
        return (CAODE, ())

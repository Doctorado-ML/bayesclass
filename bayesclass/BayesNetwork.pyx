# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp.string cimport string


cdef extern from "Network.h" namespace "bayesnet":
    cdef cppclass Network:
        Network(float, float) except + 
        void fit(vector[vector[int]]&, vector[int]&, vector[string]&, string)
        vector[int] predict(vector[vector[int]]&)
        vector[vector[double]] predict_proba(vector[vector[int]]&)
        float score(const vector[vector[int]]&, const vector[int]&)
        void addNode(string, int);
        void addEdge(string, string);
        vector[string] getFeatures();
        int getClassNumStates();
        string getClassName();
        string version()
        
cdef class BayesNetwork:
    cdef Network *thisptr
    def __cinit__(self, maxThreads=0.8, laplaceSmooth=1.0):
        self.thisptr = new Network(maxThreads, laplaceSmooth) 
    def __dealloc__(self):
        del self.thisptr
    def fit(self, X, y, features, className):
        features_bytes = [x.encode() for x in features]
        self.thisptr.fit(X, y, features_bytes, className.encode())
        return self
    def predict(self, X):
        return self.thisptr.predict(X)
    def predict_proba(self, X):
        return self.thisptr.predict_proba(X)
    def score(self, X, y):
        return self.thisptr.score(X, y)
    def addNode(self, name, states):
        self.thisptr.addNode(str.encode(name), states)
    def addEdge(self, source, destination):
        self.thisptr.addEdge(str.encode(source), str.encode(destination))
    def getFeatures(self):
        res = self.thisptr.getFeatures()
        return [x.decode() for x in res]
    def getClassName(self):
        return self.thisptr.getClassName().decode()
    def getClassNumStates(self):
        return self.thisptr.getClassNumStates()
    def __reduce__(self):
        return (BayesNetwork, ())

# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "FeatureSelect.h" namespace "features":
    ctypedef float precision_t
    cdef cppclass SelectKBestWeighted:
        SelectKBestWeighted(vector[vector[int]]&, vector[int]&, vector[precision_t]&, int, bool) except + 
        void fit()
        string version()
        vector[precision_t] getScores()
        vector[int] getFeatures()
        
cdef class CSelectKBestWeighted:
    cdef SelectKBestWeighted *thisptr
    def __cinit__(self, X, y, weights, k, natural=False): # log or log2
        self.thisptr = new SelectKBestWeighted(X, y, weights, k, natural) 
    def __dealloc__(self):
        del self.thisptr
    def fit(self,):
        self.thisptr.fit()
        return self
    def get_scores(self):
        return self.thisptr.getScores()
    def get_features(self):
        return self.thisptr.getFeatures()
    def get_version(self):
        return self.thisptr.version()
    def __reduce__(self):
        return (CSelectKBestWeighted, ())

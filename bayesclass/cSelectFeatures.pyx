# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "FeatureSelect.h" namespace "features":
    ctypedef double precision_t
    cdef cppclass SelectKBestWeighted:
        SelectKBestWeighted(vector[vector[int]]&, vector[int]&, vector[precision_t]&, int, bool) except + 
        void fit()
        string version()
        vector[precision_t] getScore()
        
cdef class CSelectKBestWeighted:
    cdef SelectKBestWeighted *thisptr
    def __cinit__(self, X, y, weights, k, natural=False): # log or log2
        self.thisptr = new SelectKBestWeighted(X, y, weights, k, natural) 
    def __dealloc__(self):
        del self.thisptr
    def fit(self,):
        self.thisptr.fit()
        return self
    def get_score(self):
        return self.thisptr.getScore()
    def get_version(self):
        return self.thisptr.version()
    def __reduce__(self):
        return (CSelectKBestWeighted, ())

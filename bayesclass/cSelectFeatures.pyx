# distutils: language = c++
# cython: language_level = 3
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "FeatureTest.h" namespace "featuresTest":
    ctypedef float precision_t
    cdef cppclass SelectKBest:
        SelectKBest(vector[int]&) except + 
        void fit()
        string version()
        
cdef class CSelectKBest:
    cdef SelectKBest *thisptr
    def __cinit__(self, X):
        self.thisptr = new SelectKBest(X)
    def __dealloc__(self):
        del self.thisptr
    def fit(self,):
        self.thisptr.fit()
        return self
    def get_version(self):
        return self.thisptr.version()
    def __reduce__(self):
        return (CSelectKBest, ())

# cdef extern from "FeatureSelect.h" namespace "features":
#     ctypedef float precision_t
#     cdef cppclass SelectKBestWeighted:
#         SelectKBestWeighted(vector[int]&) except + 
#         # SelectKBestWeighted(vector[int]&, vector[int]&, vector[precision_t]&, int) except + 
#         void fit()
#         string version()
#         vector[precision_t] getScore()
        
# cdef class CSelectKBestWeighted:
#     cdef SelectKBestWeighted *thisptr
#     def __cinit__(self, X, y, weights, k):
#         # self.thisptr = new SelectKBestWeighted(X, y, weights, k)
#         self.thisptr = new SelectKBestWeighted(X)
#     def __dealloc__(self):
#         del self.thisptr
#     def fit(self,):
#         self.thisptr.fit()
#         return self
#     def get_score(self):
#         return self.thisptr.getScore()
#     def get_version(self):
#         return self.thisptr.version()
#     def __reduce__(self):
#         return (CSelectKBestWeighted, ())

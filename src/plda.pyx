## Usage: To create an object, use /home/jfeng/workspace/plda/testdata/lda_model.txt, 0.1, 0.01, 20, 10, 123
##

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "LDA_infer.h" namespace "plda_namespace":
    cdef cppclass LDA_infer:
        LDA_infer(string, double, double, int, int, int) except +
        vector[double] run(string)

cdef class PyLDA:
    cdef LDA_infer *thisptr

    def __cinit__(self, string model_file, alpha, beta, total_iterations_, burnin_iterations_, seed):
        self.thisptr = new LDA_infer(model_file, alpha, beta, total_iterations_, 
                      burnin_iterations_, seed)

    def __dealloc__(self):
        del self.thisptr

    def run(self, string line):
        if len(line) == 0:
            return None
        return self.thisptr.run(line)

    def run_on_list(self, word_list):
        from collections import defaultdict

        store = defaultdict(int)
        for word in word_list:
            store[word] += 1

        return self.run(' '.join(("%s %s" % (k, v) for k, v in store.iteritems())))


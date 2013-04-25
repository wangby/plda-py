## Usage: To create an object, use /home/jfeng/workspace/plda/testdata/lda_model.txt, 0.1, 0.01, 20, 10, 123
##

from libcpp.string cimport string

cdef extern from "LDA_infer.h" namespace "plda_namespace":
    cdef cppclass LDA_infer:
        LDA_infer(string, double, double, int, int, int) except +
        string run(string)

cdef class PyLDA:
    cdef LDA_infer *thisptr

    def __cinit__(self, string model_file, alpha, beta, total_iterations_, burnin_iterations_, seed):
        self.thisptr = new LDA_infer(model_file, alpha, beta, total_iterations_, 
                      burnin_iterations_, seed)

    def __dealloc__(self):
        del self.thisptr

    def run(self, string line):
        return <string> self.thisptr.run(line)


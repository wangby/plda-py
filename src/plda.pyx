## Usage: To create an object, use /home/jfeng/workspace/plda/testdata/lda_model.txt, 0.1, 0.01, 20, 10, 123
##

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map

cdef extern from "LDA_infer.h" namespace "plda_namespace":
    cdef cppclass LDA_infer:
        LDA_infer(string, double, double, int, int, int) except +
        vector[double] run(string)
        map[string, double] get_related_words(vector[double], int)

cdef class PyLDA:
    """
    Wrap a trained plda model.

    Supports inference on a document and fetching the most related
    words to a topic distribution.
    """
    cdef LDA_infer *thisptr

    def __cinit__(self, string model_file,
        alpha, beta, total_iterations_, burnin_iterations_, seed):
        """Initialize the LDA model.
        model_file = a previously trained model
        alpha, beta = hyperparameters.  Use the same ones that were
            specified when training the model
        total_iterations_, burnin_iterations_ = for sampling the topics
        seed = a specified seed, or -1 to use a random seed"""
        self.thisptr = new LDA_infer(model_file,
            alpha, beta, total_iterations_, burnin_iterations_, seed)

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

    def get_related_words(self, topics, N):
        """Get most related words to these topics.

        topics is the topics distribution (vector of double, as returned from
            run)
        N = get this many words
        """
        return self.thisptr.get_related_words(topics, N)

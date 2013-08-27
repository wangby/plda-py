import unittest
import plda 


class TestPlda(unittest.TestCase):
    test_data = ["concept", "consider", "global", "entropy", "go", "contributions", "excludes", "depend", "graph",
        "environment", "program", "under", "undirected", "random", "very", "putting", "difference", "entire",
        "randomness", "july", "large", "vector", "synapses", "zl", "upper", "smaller", "says", "occurrence", "val-",
        "likely", "n", "ues", "what", "selected", "nand", "find", "access", "version", "goes", "obvious",
        "learn", "here", "desired", "objects", "let", "represented", "strong", "appears", "equiv-", "institute",
        "k", "vectors", "reports", "amount", "extremes", "proof", "regardless", "projection", "merely",
        "boolean", "total", "asymptotic", "would", "prove", "next", "automata", "taken", "tell", "knows",
        "becomes", "visual", "appendix", "normalized", "particular", "hold", "must", "work", "itself",
        "values", "v", "abu-mostafa", "process", "sample", "something", "arise", "distinguishable", "occur", "huge",
        "end", "rather", "means", "feature", "write", "infor-", "spon-", "ensemble", "information", "may", "after",
        "consequence", "designed", "en-", "complexity", "so", "sb", "restriction", "holds", "office", "produces",
        "yaser", "paper", "through", "ity", "still", "denker", "symmetry", "how", "coordinates", "distinguishing",
        "systems", "main", "versus", "eventually", "imple-", "synapse", "introduce", "thus", "now", "nor", "term",
        "subset", "el", "doing", "ea", "idea", "frequency"]

    def approx_equal(self, num1, num2, error=0.05):
        return abs(num1 - num2) <= 0.5 * error * (num1 + num2)

    def test_accuracy_string(self):
        # This is 10 records we got from the test-data set in plda package.
        test_data_file = 'data/test_data_10l.txt'

        # This is the data we got by running command line infer tool in the plda package.
        expected = [
            [114.05, 273.95],
            [440.03, 439.97],
            [124.72, 386.28],
            [266.75, 350.25],
            [302.31, 558.69],
            [131.61, 401.39],
            [529.27, 256.73],
            [141.16, 394.84],
            [542.17, 179.83],
            [271.04, 344.96]
        ]

        model_file = 'data/lda_model.txt'

        alpha = 0.1
        beta = 0.01
        max_iter = 200
        burnin_iter = 100
        seed = -1   # use the original seed setting (time) in plda package.

        model = plda.PyLDA(model_file, alpha, beta, max_iter, burnin_iter, seed)

        with open(test_data_file) as fin:
            i = 0
            for line in fin:
                calculated = model.run(line)

                # Due to the randomness of the plda algorithm, the values may differ by about 5%.
                for j in xrange(len(calculated)):
                    self.assertTrue(self.approx_equal(calculated[j], expected[i][j]))

                i += 1

    def test_accuracy_list(self):
        # This is 10 records we got from the test-data set in plda package.

        # This is the data we got by running command line infer tool in the plda package.
        expected = [46.21, 89.79]

        model_file = 'data/lda_model.txt'

        alpha = 0.1
        beta = 0.01
        max_iter = 200
        burnin_iter = 100
        seed = -1   # use the original seed setting (time) in plda package.

        model = plda.PyLDA(model_file, alpha, beta, max_iter, burnin_iter, seed)

        calculated = model.run_on_list(TestPlda.test_data)

        # Due to the randomness of the plda algorithm, the values may differ by about 5%.
        for j in xrange(len(calculated)):
            self.assertTrue(self.approx_equal(calculated[j], expected[j]))

    def test_accuracy_list_unicode(self):
        model_file = 'data/lda_model.txt'

        test_data_unicode = TestPlda.test_data + [u'registered\xae']
        expected = [46.21, 89.79]

        alpha = 0.1
        beta = 0.01
        max_iter = 200
        burnin_iter = 100
        seed = -1   # use the original seed setting (time) in plda package.

        model = plda.PyLDA(model_file, alpha, beta, max_iter, burnin_iter, seed)

        calculated = model.run_on_list(test_data_unicode)

        # Due to the randomness of the plda algorithm, the values may differ by about 5%.
        for j in xrange(len(calculated)):
            self.assertTrue(self.approx_equal(calculated[j], expected[j]))

    def test_memory_leak(self):
        """
        This test case is used to test if there exists memory leak in the c++ codes by repeatedly calling model.run
        method.
        :return:
        """
        with open("data/test_data_1l.txt", "r") as fin:
            test_data=fin.read()

        # This is the data we got by running command line infer tool in the plda package.
        expected = [440.03, 439.97]

        model_file = 'data/lda_model.txt'

        alpha = 0.1
        beta = 0.01
        max_iter = 200
        burnin_iter = 100
        seed = -1   # use the original seed setting (time) in plda package.

        model = plda.PyLDA(model_file, alpha, beta, max_iter, burnin_iter, seed)

        for i in xrange(1):  # 10000 runs should take about 10 minutes to complete.
            calculated = model.run(test_data)

            for j in xrange(len(calculated)):
                self.assertTrue(self.approx_equal(calculated[j], expected[j], error=0.1))


if __name__ == "__main__":
    unittest.main()

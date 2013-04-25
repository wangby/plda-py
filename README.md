plda
=======

Python interface for using plda C++ program


The core C++ codes for plda is from http://code.google.com/p/plda/

The Python wrapper for plda downloaded the plda codes from http://code.google.com/p/plda/downloads/detail?name=plda-3.1.tar.gz&can=2&q=

The added `LDA_infer.cc` and `LDA_infer.h` define a C++ class that wraps the inference functions in the original codes into a class. And the `plda.pyx` and the `setup.py` are created to use Cython to build the python module. 

To build the wrapper, run `python setup.py build`.
To install the wrapper, run `(sudo) python setup.py install`.

To run the inference with a model in Python, use codes like the following,
`
    import plda

    model_file = './testdata/lda_model.txt' # this file contains the model you trained using lda command.

    alpha = 0.1
    beta = 0.01
    max_iter = 200
    burnin_iter = 100
    seed = -1

    model = plda.PyLDA(model_file, alpha, beta, max_iter, burnin_iter, seed) # create a model object from the model file.
    result = model.run(line) # the line is a record of data.
`

The package is able to compile, install and run with Python 2.7.3, Cython 0.18 in Ubuntu 12.04 LTS 64b.

This module should be able to support the future versions of plda. The only things needed to do for updating are,
1. Download the new C++ codes into the 'plda' folder.
2. Make a copy of the file `infer.cc` and name it to `LDA_infer.cc`. Modify it to make it a class definition. Then Create a .h file for the class definition.

JFeng, SEOmoz, 04/2013

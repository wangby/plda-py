#!/usr/bin/env python

# Copyright (c) 2013 SEOmoz
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_modules = [
    Extension("plda", 
    sources = ["src/plda.pyx", "plda/accumulative_model.cc", "src/LDA_infer.cc", "plda/common.cc", "plda/document.cc", "plda/model.cc", "plda/cmd_flags.cc", "plda/sampler.cc"], 
    language="c++")]

setup(
    name             = 'plda-py',
    version          = '0.1.0',
    description      = 'Python wrapper from SEOmoz for plda (http://code.google.com/p/plda)',
    author           = 'Jerry(Jian) Feng', 
    author_email     = 'jerry@seomoz.org',
    url              = 'http://github.com/seomoz/plda-py',
    license          = 'MIT',
    platforms        = 'Posix; MacOS X',
    cmdclass         = {'build_ext': build_ext},
    ext_modules      = ext_modules,
    classifiers      = [
        'License :: OSI Approved :: MIT License',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research'
        ],
)


from setuptools import setup, find_packages
import codecs
import os
import re

def read(fname):
    with codecs.open(fname, 'r', 'latin') as f:
        return f.read()
    
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with codecs.open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()
    

def get_version():
    VERSIONFILE = os.path.join('pymcmcstat', '__init__.py')
    with open(VERSIONFILE, 'rt') as f:
        lines = f.readlines()
    vgx = '^__version__ = \"\d+\.\d+\.\d.*\"'
    for line in lines:
        mo = re.search(vgx, line, re.M)
        if mo:
            return mo.group().split('"')[1]
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))

setup(
    name='pymcmcstat',
    version=get_version(),
    description=('A library to perform MCMC simulations using DRAM'),
    long_description=long_description,
    url='https://github.com/prmiles/pymcmcstat',
    download_url='https://github.com/prmiles/pymcmcstat',
    author='Paul Miles',
    author_email='prmiles@ncsu.edu',
    license='MIT',
    package_dir={'pymcmcstat': 'pymcmcstat'},
    packages=find_packages(),
    zip_safe=False,
    install_requires=['numpy>=1.14', 'scipy>=1.0', 'matplotlib>=2.2.0', 'h5py>=2.7.0', 'statsmodels>=0.9.0'],
    extras_require = {'docs':['sphinx'], 'plotting':['matplotlib', 'plotly'],},
    classifiers=['License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6',
                   'Framework :: IPython',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering',
                   ]
)

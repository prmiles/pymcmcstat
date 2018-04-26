from setuptools import setup
import codecs

def read(fname):
    with codecs.open(fname, 'r', 'latin') as f:
        return f.read()
    
# read in version number
version_dummy = {}
exec(read('pymcmcstat/__version__.py'), version_dummy)
__version__ = version_dummy['__version__']
del version_dummy

setup(
    name='pymcmcstat',
    version=__version__,
    description='A library to perform mcmc simulations',
    url='https://github.com/prmiles/pymcmcstat',
    author='Paul Miles',
    author_email='prmiles@ncsu.edu',
    license='MIT',
    packages=['pymcmcstat'],
    dependency_links=['http://github.com/prmiles/pymcmcstat/tarball/master#egg=package-1.0'],
    zip_safe=False,
    install_requires=['numpy','scipy']
)
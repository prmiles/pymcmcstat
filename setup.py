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
    download_url='https://github.com/prmiles/pymcmcstat',
    author='Paul Miles',
    author_email='prmiles@ncsu.edu',
    license='MIT',
    package_dir={'pymcmcstat': 'pymcmcstat'},
    packages=['pymcmcstat',
              'pymcmcstat.chain',
              'pymcmcstat.plotting',
              'pymcmcstat.procedures',
              'pymcmcstat.samplers',
              'pymcmcstat.settings',
              'pymcmcstat.structures',
              'pymcmcstat.utilities'],
#    dependency_links=['http://github.com/prmiles/pymcmcstat/tarball/master#egg=package-1.0'],
    zip_safe=False,
    install_requires=['numpy>=1.7', 'scipy>=0.16', 'matplotlib', 'h5py', 'statsmodels'],
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

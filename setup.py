from setuptools import setup
setup(
    name='pymcmcstat',
    version='1.3.1',
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

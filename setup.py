from setuptools import setup
setup(
    name='pymcmcstat',
    version='0.2',
    description='A library to perform mcmc simulations',
    url='https://github.com/prmiles/pymcmcstat',
    author='Paul Miles',
    author_email='prmiles@ncsu.edu',
    license='MIT',
    packages=['pymcmcstat'],
    zip_safe=False,
    install_requires=[
          'sys',
          'time',
          'numpy',
          'math',
          'matplotlib',
          'datetime',
      ],
)
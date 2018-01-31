#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:12:50 2018

@author: prmiles
"""
# import required packages
import numpy as np

class ErrorVarianceEstimator:
    
    def update_error_variance(self, sos, model):
        N0 = model.N0
        S20 = model.S20
        N = model.N
        sigma2 = model.sigma2 # initializes it as array
        nsos = len(sos)
        
        for jj in range(0,nsos):
            sigma2[jj] = (self.__gammar(1, 1, 0.5*(N0[jj]+N[jj]),
                          2*((N0[jj]*S20[jj]+sos[jj])**(-1))))**(-1)

        return sigma2
            
    def __gammar(self, m,n,a,b = 1):
        #%GAMMAR random deviates from gamma distribution
        #%  GAMMAR(M,N,A,B) returns a M*N matrix of random deviates from the Gamma
        #%  distribution with shape parameter A and scale parameter B:
        #%
        #%  p(x|A,B) = B^-A/gamma(A)*x^(A-1)*exp(-x/B)
        #
        #% Marko Laine <Marko.Laine@Helsinki.FI>
        #% Written for python by PRM
        
        if a <= 0: # special case
            y = np.zeros([m,n])
            return y
        
        y = self.__gammar_mt(m, n, a, b)
        return y
    
    def __gammar_mt(self, m, n, a, b = 1):
        #%GAMMAR_MT random deviates from gamma distribution
        #% 
        #%  GAMMAR_MT(M,N,A,B) returns a M*N matrix of random deviates from the Gamma
        #%  distribution with shape parameter A and scale parameter B:
        #%
        #%  p(x|A,B) = B^-A/gamma(A)*x^(A-1)*exp(-x/B)
        #%
        #%  Uses method of Marsaglia and Tsang (2000)
        #
        #% G. Marsaglia and W. W. Tsang:
        #% A Simple Method for Generating Gamma Variables,
        #% ACM Transactions on Mathematical Software, Vol. 26, No. 3,
        #% September 2000, 363-372.
        # Written for python by PRM
        import numpy as np
        y = np.zeros([m,n])
        for jj in range(0,n):
            for ii in range(0,m):
                y[ii,jj] = self.__gammar_mt1(a,b)
                
        return y
        
    def __gammar_mt1(self, a,b):
        if a < 1:
            y = self.__gammar_mt1(1+a,b)*np.random.rand(1)**(a**(-1))
            return y
        else:
            d = a - 3**(-1)
            c = (9*d)**(-0.5)
            while 1:
                while 1:
                    x = np.random.randn(1)
                    v = 1 + c*x
                    if v > 0:
                        break
                    
                v = v**(3)
                u = np.random.rand(1)
                if u < 1-0.0331*x**(4):
                    break
                if np.log(u) < 0.5*x**2 + d*(1-v+np.log(v)):
                    break
            
            y = b*d*v
            return y
from numpy import genfromtxt
from pylab import *
from scipy.optimize import curve_fit
from numpy import exp, sqrt
import pandas as pd

data = genfromtxt("data/hsha/sc_se_latency")

base_count = 25
multiplier = 1.0
bins = np.arange(base_count, step=(1 / multiplier))
print (bins)
y,x,_=hist(data,bins=bins,alpha=.3,label='data')

x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def trimodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)

expected=(10,1,91000,13,1,152000,16,1,65000)
params,cov=curve_fit(trimodal,x,y,expected)
sigma=sqrt(diag(cov))
plot(x,trimodal(x,*params),color='red',lw=3,label='model')
legend()
#print('Result ', params,'\n',sigma) 
print (pd.DataFrame(data={'params':params,'sigma':sigma},index=trimodal.__code__.co_varnames[1:]))
show()
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:43:05 2018

@author: kalas
"""


from pylab import *
#from scipy import optimize
from sympy.solvers import solve
from sympy import Symbol, re
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import savgol_filter
import csv
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.gridspec as gridspec
import tvregdiff
from scipy.optimize import curve_fit
import math
import JJformulas


#import MBlib

fdir = 'E:\\OneDrive - Rutgers University\\files_py\\expdata\\'
#fname = [ '4p05','3p69' ,'3p88']
#fname = [ '180819-68N5_IVFFf150mK-011019.txt']


file = '781T_IVZFl35m_IswIstep_bat-173155.txt'

data = np.genfromtxt(fdir + file, skip_header = 22 ) [1:,:] 

T = data[:,5]
Iraw = data[:,7]
Vraw = data[:,8] 

Isw = np.array([])
Ir = np.array([])

Iarr = np.array([])


vjump = 2e-4

I = Iraw[:-1]
V = Vraw[:-1]

dV = np.diff(Vraw)
dI = np.diff(Iraw)


step = 4.0e-11

eps = 1e-20

I_iv = I[ abs(abs(dI) - step ) < eps   ]
V_iv = V[ abs(abs(dI) - step ) < eps   ]

Isw = I [ dV > vjump] 
Vsw = V [ dV > vjump]
dIsw = abs(dI [ dV > vjump]) 
 

Ir = I [ dV < -vjump] 
Vr = V [ dV < -vjump] 
dIr = abs(dI [ dV < -vjump]) 


plt.plot(Iraw,Vraw, '.-')
plt.plot(Isw,Vsw, 'rx')
plt.plot(Ir,Vr, 'kx')

plt.subplots()
plt.plot(dIr,Ir, 'kx')
plt.plot(dIsw,Isw, 'rx')

plt.subplots()
plt.plot(I_iv,V_iv, '-x')


#####data generating#####
Imax = 3000

stMax = 30
stMin = 14

#stepArr = np.linspace(stMin, stMax, round((stMax-stMin) / 2e-12  )) 
stepArr = np.array([14, 16, 18, 20, 25, 30])*3


for st in stepArr:
    nstep = round(Imax/st)
    
    Iarr = np.append(Iarr, np.linspace(0, nstep*st, nstep+1) )
    Iarr = np.append(Iarr, np.linspace( nstep*st,0, nstep+1) )
    


Iarr_str = []    

for i in Iarr:
    
    Iarr_str = append(Iarr_str, '%3.5e' % (i*1e-12))
    
wfname  = fdir + 'IswVarStep' + '%2.0f' % ( stepArr[0]) +'t'+'%2.0f' % (stepArr[-1]) + '_'+ '%1.0f' % (len(stepArr))+'d.txt'
    
with open(wfname, "w") as otp:
            
   writer = csv.writer(otp, delimiter= '\n', lineterminator='\n')
            
   writer.writerow(Iarr_str )
            
   otp.close()
    
    
    
    

 

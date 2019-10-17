# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:46:26 2018

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
import JJformulas


#import MBlib

fdir = 'E:\\OneDrive - Rutgers University\\files_py\\expdata\\'
#fname = [ '4p05','3p69' ,'3p88']
#fname = [ '180819-68N5_IVFFf150mK-011019.txt']


Frust = 'IV scanB'
spl = 'chain 68N5\\' + Frust + '\\'


def avg_group(vA0, vB0):
    vA0 = np.round(vA0*1e15)/1e15   #remove small deferences
    vB0 = np.round(vB0*1e15)/1e15
    
    vA, ind, counts = np.unique(vA0, return_index=True, return_counts=True) # get unique values in vA0
    vB = vB0[ind]
    for dup in vB[counts>1]: # store the average (one may change as wished) of original elements in vA0 reference by the unique elements in vB
        vB[np.where(vA==dup)] = np.average(vB0[np.where(vA0==dup)])
    return vA, vB


def V_func(I,V, val):
    out = []
    for x in np.nditer(val):
        out = np.append (out,  V[np.argmin(abs(I-x))])
    return out


def diffArr(Xarr, Yarr, step):
    out = []
    
    for x in Xarr:
        out = np.append(out, (V_func(Xarr,Yarr, x+step/2)  - V_func(Xarr,Yarr, x-step/2))/(step))
    return out


def R0byFit (I,V,n = 3):
    V = np.append(V, V[-n:])
    I = np.append(I, I[-n:])
    
    out = []
    
    for i in range(len(I)-n):    
        a, b = polyfit (I [i:i+n] , V [i:i+n], 1 )
        out = np.append(out, a)
        
    return out

def XYEqSp(Xarr, Yarr, step):
    outX = []
    outY = []

    n = int((np.max(Xarr) - np.min(Xarr)) // step)    
    
    for i in range(n):
        outX = np.append( outX, V_func(Xarr, Xarr, np.min(Xarr) + i*step)  )
        outY = np.append( outY, V_func(Xarr, Yarr, np.min(Xarr) + i*step)  )

    return outX, outY


def offsetRemove(I,V, Istep, mode = 'ZF'):
#    plt.figure()
       
    Rdiff = Rdiff_TVReg(V, Istep)
    
    ind_minR = np.argmin(Rdiff)
    ind_maxR = np.argmax(Rdiff)


    if mode == 'ZF':
        Ioff = I[ind_minR]
    else:
        Ioff = I[ind_maxR]

    
    Voff = V_func(I, V, Ioff)

    Inew = I - Ioff
    Vnew = V - Voff
    
#    plt.plot(abs(I), abs(V), '.')
#    plt.plot(abs(Inew), abs(Vnew), '-')
    return Inew, Vnew

def Rdiff_TVReg(V, Istep):
    stepx = 0.05
    Rdiff = (TVRegDiff(V, 100, 7e-2, dx = stepx, ep=1e-1, scale='small', plotflag=0)*stepx / Istep)[:-1]
    return Rdiff


for f in os.listdir(fdir + spl):
    if f.startswith('18'):
       
        os.rename( os.path.join(fdir + spl, f),  os.path.join(fdir + spl, f[7:]))


Tarr = np.array([])
R0arr = np.array([])
Isw =  np.array([])
Rmax =  np.array([])
Rmin =  np.array([])



IcArr =  np.array([])
gamArr =  np.array([])



figsize = (14, 30)
cols = 2

fig1, ax1 = plt.subplots()

Ej = 1.52


for i, f in enumerate(os.listdir(fdir + spl)[0:1]):


    I, V = [], []
    
    data = np.genfromtxt(fdir + spl+f, skip_header = 22 ) [1:,:] 

    dataToSort = data[:, 5:9]
    

    IVbyB = [ dataToSort[ dataToSort[:,1] == k ] for k in np.unique(dataToSort[:,1])]

    nf = 2
    
    for i, IV in enumerate(IVbyB[nf:nf+1]):
        
        Iraw = IV[:,2]
        Vraw = IV[:,3]
        B = np.mean(IV[:,1])


        T = data[:,5]


        Istep = 100e-12
     
        R_sl = slice(5,-5)
    
        if Frust == 'IV ZF' or 'IV scanB':
            ind = abs(Vraw) < (40e-6 *1)
            Iavg, Vavg = avg_group(Iraw[ind], Vraw[ind])
    
            I_eqst, V_eqst = XYEqSp(Iavg, Vavg, Istep)
            I, V = offsetRemove(I_eqst, V_eqst, Istep)   
    
            Rdiff = Rdiff_TVReg(V, Istep)[R_sl]
            I_R = I[R_sl]
            
        else:
            Iavg, Vavg = avg_group(Iraw, Vraw)
            I, V = offsetRemove(Iavg, Vavg, Istep, mode = 'FF')   
        
            Rdiff = np.diff(V)/np.diff(I)
            Rdiff = np.append(Rdiff, Rdiff[-1])[R_sl]
            I_R = I[R_sl]
        

    
         


    

        ax1.set_title('%1.0f mK' % mean(T*1e3) )
        ax1.plot(I*1e9, V*10e3, '-')    
    
        ax1.set_xlabel('I (nA)')
        ax1.set_ylabel('V (mV)')




    
#        ax2 = ax1.twinx()
##
##    
#        ax2.plot(I_R*1e9,Rdiff/1e3, 'r', label = 'exp $R_0$ = %1.1f kOhm' % mean((Rdiff[np.argmin(abs(I_R))])/1e3) )
#
#    
#        ax2.set_yscale('log')
#        ax2.set_ylabel('R (kOhm)')

    

#    ax2.set_xlim (-1e-10,1e-10)
#    ax1.set_ylim (-0.7e-5,-0.2e-5)
    
        gam = 5
    
        Rn = 66e5
        Ic = np.max(abs(I))   
        
        h = 6.64e-34
        e = 1.6e-19
    
        Ic = 7.0e-9
        T = 0.3
    #        Ej = 0.8
        Ec = 1
        Q = 1
        Rn = 10e3    
        Ej = 1       
            
        
#        V_fit = lambda I_var, Ic, gam : V_PD (I_var/Ic, Ej/gam , Ej , Ec , Q , Rn, mode = 'AH'  )           
#        V_AmbHal = lambda x, Ic,gam : Ic*Rn*2*(1-(x/Ic)**2)**0.5*np.exp(-gam*((1-(x/Ic)**2)**0.5 + (x/Ic)*np.arcsin((x/Ic))))*np.sinh(0.5*3.14*gam*(x/Ic))
#        R_AmbHal = lambda x, Ic,gam : -V_AmbHal(x, Ic, gam)/Ic*gam*np.sinh(x/Ic) - x/Ic*V_AmbHal(x, Ic, gam)/Ic/(1 - (x/Ic)**2)**0.5 + Rn*3.14*gam*(1-(x/Ic)**2)**0.5*np.exp(-gam*((1-(x/Ic)**2)**0.5 + (x/Ic)*np.arcsin((x/Ic))))*np.cosh(0.5*3.14*gam*(x/Ic))   
        V_fit = lambda I_var, Ic, gam : V_Likh (I_var, Ic , Ej/gam  ,Ej,  Rn  )
        
        
        popt1 = popt = (5e-9, 5)
        
    

        if Frust == 'IV ZF' or 'IV scanB':
         for i in range(1):
    
            
    
            popt, pcov = curve_fit(V_fit, I, V, p0 = popt1 )
        
#            try:
#                popt1, pcov1 = curve_fit(R_AmbHal, I_R, Rdiff, p0 = popt)
#            except RuntimeError:
#                print ('tadam')
#                popt1 = popt
#        
        
#        V_fit = V_AmbHal(I,  *popt)
#        R_fit = np.diff(V_fit)/np.diff(I)
#        
#        print (popt)
#        print (popt1)



    
            ax1.plot(I*1e9, V_fit(I,  *popt)*10e3, 'b.',label = ' $I_1$ = %1.1f nA, $T_{eff}$ = %1.2f K' % (popt[0]*1e9, Ej/popt[1]))
#            ax1.plot(I*1e9, V_AmbHal(I,  *popt1)*10e3, 'rx')
    
    
         
#            ax2.plot(I*1e9,R_AmbHal(I, *popt)/1e3,'b.', label = 'fit to IV; Ic = %1.1f nA, $T_{eff}$ = %1.0f K' % (popt[0]*1e9, 1400/popt[1]) )
#    
#            ax2.plot(I*1e9,R_AmbHal(I, *popt1)/1e3,'rx', label = 'fit to IR; Ic = %1.1f nA, $T_{eff}$ = %1.0f K' % (popt1[0]*1e9, 1400/popt1[1]))
    
    
#            ax2.set_yticks([10])

            plt.legend()
    

    
#    Tarr = np.append(Tarr, np.mean(T*1e3) )
#    R0arr = np.append(R0arr, Rdiff[np.argmin(abs(I_R))] )
#    Rmax = np.append(Rmax, np.max(Rdiff) )
#    Rmin = np.append(Rmin, np.min(Rdiff) )
#
#
#    Isw = np.append(Isw, np.max(abs(I)) )
#
#    IcArr = np.append(IcArr, popt1[0] )
# 
#    gamArr = np.append(gamArr, popt1[1] )

    
#    print (I[np.argmin(abs(Rdiff))])
#    
#    V_noise = []
#    for i in I:
#        V_noise = np.append (V_noise, np.mean( V_func(I,V, i + 10e-11*sin(np.linspace(-3.14, 3.14, num = 100) ) )))
    
    


#    
#fig,ax1 = plt.subplots()    
#
#ax1.plot(Tarr,IcArr*1e9, 'b-x', label = 'Ic(T)')  
#
#ax1.set_xlabel('T (mK)')
#ax1.set_ylabel('Ic (nA)')
#
#
#
#ax2 = ax1.twinx()
#ax2.plot(Tarr, Isw*1e9, 'r-x', label = 'Isw(T)') 
#ax2.set_ylabel('Isw (nA)')
#   
#
#
##ax1.set_yscale('log')
#ax1.legend()
#ax2.legend()  
#
#
#fig,ax1 = plt.subplots()    
#
#   
#ax1.plot(Tarr[:-1],Ej*1e3/gamArr[:-1], 'r-x', label = '$T_{eff}$($T_{ph}$)')
#
#ax1.set_xlabel('$T_{ph}(mK)$')
#ax1.set_ylabel('$T_{eff}(mK)$')
#
##ax1.set_yscale('log')
#ax1.legend()
#  
#
#
#  
#  
#    
#
#
#    
#fig,ax1 = plt.subplots()    
#ax1.plot(1e3/Tarr,R0arr/1e3, 'b-x', label = 'R0')  
##   
##ax1.plot(1/Tarr,Rmax, 'r-x', label = 'R_mmin')
#
#ax1.set_xlabel('1/T ($mK^{-1}$)')
#ax1.set_ylabel('R0 (kOhm)')
#
#ax1.set_yscale('log')
#ax1.legend()
#
##  
##fig,ax1 = plt.subplots()    
##ax1.plot(Tarr,Isw, 'b-x') 
##ax1.set_yscale('log')
##
###fig,ax1 = plt.subplots()    
###ax1.plot(I,V_noise, 'b-x') 
##
##a, b = polyfit (1/Tarr [-4:] , np.log (R0arr [-4:]), 1 )
##
##print ('R0 %1.2f' %(a/1e3))
##
##fig,ax1 = plt.subplots()
##ax1.plot(1/Tarr[-6:],np.log (R0arr [-6:]), 'rx', 1/Tarr[-6:], a*1/Tarr[-6:] + b, 'b-', label = 'R0')  
##
##
##a, b = polyfit (1/Tarr [-6:] , np.log (Rmax [-6:]), 1 )
##
##print ('Rmax %1.2f' % (a/1e3) )
##
##ax1.plot(1/Tarr[-6:],np.log (Rmax [-6:]), 'gx', 1/Tarr[-6:], a*1/Tarr[-6:] + b, 'k-', label = 'R_max')
##ax1.legend()
#
#
#
#    
#
#
#   
#        Rdiff = (TVRegDiff(V, 100, 5e-2, dx=0.05, ep=1e-2, scale='small', plotflag=0) / Istep)[:-1]    
#    
#        ax2.plot(I[:],Rdiff, '--', label = '%1.0f Imc' % (B/1e-5) )
#        ax2.set_yscale('log')
#
#
#        plt.legend()
#    
#   
#
#    

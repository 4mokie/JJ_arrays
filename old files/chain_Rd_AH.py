# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 18:50:12 2018

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


Frust = 'IV FF'
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
        
        
        
        out = np.append(out, np.mean((V_func(Xarr,Yarr, x+step/2))  - np.mean(V_func(Xarr,Yarr, x-step/2)))/(step))
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


def offsetRemove(I,V, Istep, mode = 'ZF', Ioff_def = 8e-12):
#    plt.figure()
       
    Rdiff = Rdiff_TVReg(V, Istep)
    
    ind_minR = np.argmin(Rdiff)
    ind_maxR = np.argmax(Rdiff)


    if mode == 'ZF':
        Ioff = I[ind_minR]
    else:
        Ioff = Ioff_def

    
    Voff = V_func(I, V, Ioff)

    Inew = I - Ioff
    Vnew = V - Voff
    
#    plt.plot(abs(I), abs(V), '.')
#    plt.plot(abs(Inew), abs(Vnew), '-')
    return Inew, Vnew

def Rdiff_TVReg(V, Istep):
    stepx = 0.05
    Rdiff = (TVRegDiff(V, 100, 10e-3, dx = stepx, ep=1e-1, scale='small', plotflag=0)*stepx / Istep)[:-1]
    return Rdiff

def eng_string( x, sig_figs=3, si=True):
    x = float(x)
    sign = ''
    if x < 0:
        x = -x
        sign = '-'
    if x == 0:
        exp = 0
        exp3 = 0
        x3 = 0
    else:
        exp = int(math.floor(math.log10( x )))
        exp3 = exp - ( exp % 3)
        x3 = x / ( 10 ** exp3)
        x3 = round( x3, -int( math.floor(math.log10( x3 )) - (sig_figs-1)) )
        if x3 == int(x3): # prevent from displaying .0
            x3 = int(x3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = 'yzafpnum kMGTPEZY'[ exp3 // 3 + 8]
    elif exp3 == 0:
        exp3_text = ''
    else:
        exp3_text = 'e%s' % exp3

    return ( '%s%s%s') % ( sign, x3, exp3_text)



for f in os.listdir(fdir + spl):
    if f.startswith('18'):
       
        os.rename( os.path.join(fdir + spl, f),  os.path.join(fdir + spl, f[7:]))


Tarr = np.array([])
R0arr = np.array([])
Isw =  np.array([])
Rmax =  np.array([])
Rmin =  np.array([])
IRmax =  np.array([])



IcArr =  np.array([])
gamArr =  np.array([])



figsize = (15,15)
cols = 3
gs = gridspec.GridSpec(len(os.listdir(fdir + spl)) // cols + 1, cols)
gs.update(hspace=0.5)
gs.update(wspace=.3)

fig1 = plt.figure(num=1, figsize=figsize)
ax1 = []
ax2 = []

Ej = 1.52*2


for i, f in enumerate(os.listdir(fdir + spl)[0:9]):


    I, V = [], []
    
    data = np.genfromtxt(fdir + spl+f, skip_header = 22 ) [1:,:] 
#    print(np.mean(data[:,5]))

    T = data[:,5]
    Iraw = data[:,7]
    Vraw = data[:,8]/20


    
 
    R_sl = slice(5,-5)

    if Frust == 'IV ZF':
        Istep = 10e-11
        
        ind = abs(Vraw) < (40e-6 *1)
        Iavg, Vavg = avg_group(Iraw[ind], Vraw[ind])

        I_eqst, V_eqst = XYEqSp(Iavg, Vavg, Istep)
        I, V = offsetRemove(I_eqst, V_eqst, Istep)   

        Rdiff = R0byFit(I,V,10)[R_sl]
        Rdiff = Rdiff_TVReg(V, Istep)[R_sl]
#        Rdiff = np.diff(V)/np.diff(I)
#        Rdiff = np.append(Rdiff, Rdiff[-1])[R_sl]

        I_R = I[R_sl]
        
    else:
        Istep = 10e-11

        ind = abs(Vraw) < (40e-6 *1)
        Iavg, Vavg = avg_group(Iraw[ind], Vraw[ind])

        
#        Iavg, Vavg = avg_group(Iraw, Vraw)

#        I_eqst, V_eqst = XYEqSp(Iavg, Vavg, Istep)

        I, V = offsetRemove(Iavg, Vavg, Istep, mode = 'FF',  Ioff_def = 18e-12)   
#        I, V = Iavg, Vavg   


    
        Rdiff = np.diff(V)/np.diff(I)
        Rdiff = np.append(Rdiff, Rdiff[-1])[R_sl]
        I_R = I[R_sl]
    

    
         
#    print(I_R[np.argmax(Rdiff)])

    

    row = (i // cols)
    col = i % cols
    ax1.append(fig1.add_subplot(gs[row, col]))

    ax1[-1].set_title('%1.0f mK' % mean(T*1e3) )
    ax1[-1].plot(I*1e9, V*10e3, '-x')    

    ax1[-1].set_xlabel('I (nA)')
    ax1[-1].set_ylabel('V (mV)')
#    ax1[-1].set_xlim(-0.25,0.25)



#    ax1[-1].plot(Iavg,Vavg, 'r-')    

#    plt.plot(Iraw,Vraw, 'g.')
#    plt.plot(Iavg,Vavg, 'r-')
  
    
#    ax1.plot(I,V, 'bx')


#    print ('T = %1.0f mK, Spacing avg %1.2f' %( mean(T*1e3) , mean(np.diff(I))/1e-12))
#    
#    if round (mean(T*1e2)) == 40:
#        print (I,V, Rdiff)

#    Rdiff = np.diff(V)/np.diff(I)
#    Rdiff = np.append(Rdiff, Rdiff[-1])
    
    
#    Rdiff = diffbyStep(V, 10)/diffbyStep(I, 10)

#    Rdiff = R0byFit(I,V, 10)


    
#    Rdiff[Rdiff > 1e9] = np.min(Rdiff)
    
    ax2 = ax1[-1].twinx()
#
#    
  
###########    
    
    ax2.plot(I_R*1e9,Rdiff, 'r', label = 'exp $R_0$ = %1.10sOhm' % eng_string((Rdiff[np.argmin(abs(I_R))])) )

##########    
    ax2.set_yscale('log')
    ax2.set_ylabel('R (Ohm)')

    

#    ax2.set_xlim (-1e-10,1e-10)
#    ax1.set_ylim (-0.7e-5,-0.2e-5)
    
    gam = 5
    
    Rn = 2e7
    Ic = np.max(abs(I))   
    
    h = 6.64e-34
    e = 1.6e-19
       
    V_AmbHal = lambda x, Ic,gam : Ic*Rn*2*(1-(x/Ic)**2)**0.5*np.exp(-gam*((1-(x/Ic)**2)**0.5 + (x/Ic)*np.arcsin((x/Ic))))*np.sinh(0.5*3.14*gam*(x/Ic))
    R_AmbHal = lambda x, Ic,gam : -V_AmbHal(x, Ic, gam)/Ic*gam*np.sinh(x/Ic) - x/Ic*V_AmbHal(x, Ic, gam)/Ic/(1 - (x/Ic)**2)**0.5 + Rn*3.14*gam*(1-(x/Ic)**2)**0.5*np.exp(-gam*((1-(x/Ic)**2)**0.5 + (x/Ic)*np.arcsin((x/Ic))))*np.cosh(0.5*3.14*gam*(x/Ic))   
    
    
    
    
    popt1 = popt = (7e-9, 4)
    
    

    if Frust == 'IV ZF':
     for i in range(1):

        

        popt, pcov = curve_fit(V_AmbHal, I, V, p0 = popt1 )
#        popt, pcov = curve_fit(V_AmbHal, I[R_sl], V[R_sl], p0 = popt1 )
    
        try:
            popt1, pcov1 = curve_fit(R_AmbHal, I_R, Rdiff, p0 = popt)
        except RuntimeError:
            print ('tadam')
            popt1 = popt
    
        
#        V_fit = V_AmbHal(I,  *popt)
#        R_fit = np.diff(V_fit)/np.diff(I)
#        
#        print (popt)
#        print (popt1)



 ################   
        ax1[-1].plot(I*1e9, V_AmbHal(I,  *popt)*10e3, 'b.')
        ax1[-1].plot(I*1e9, V_AmbHal(I,  *popt1)*10e3, 'rx')


     
        ax2.plot(I*1e9,R_AmbHal(I, *popt),'b.', label = 'fit to V(I); $I_1$ = %1.1f nA, $T_{eff}$ = %1.0f K' % (popt[0]*1e9, Ej*1e3/popt[1]) )

        ax2.plot(I*1e9,R_AmbHal(I, *popt1),'rx', label = 'fit to R(I); $I_1$ = %1.1f nA, $T_{eff}$ = %1.0f K' % (popt1[0]*1e9, Ej*1e3/popt1[1]))


        ax2.set_yticks([10])
#######################
    plt.legend()
    

    
    Tarr = np.append(Tarr, np.mean(T*1e3) )
    R0arr = np.append(R0arr, Rdiff[np.argmin(abs(I_R))] )
    Rmax = np.append(Rmax, np.max(Rdiff) )
    IRmax = np.append(IRmax, abs(I_R[np.argmax(Rdiff)]) )
    
    Rmin = np.append(Rmin, np.min(Rdiff) )


    Isw = np.append(Isw, np.max(abs(I)) )

    IcArr = np.append(IcArr, popt1[0] )
 
    gamArr = np.append(gamArr, popt1[1] )

    
#    print (I[np.argmin(abs(Rdiff))])
#    
#    V_noise = []
#    for i in I:
#        V_noise = np.append (V_noise, np.mean( V_func(I,V, i + 10e-11*sin(np.linspace(-3.14, 3.14, num = 100) ) )))
    
    


fig,ax1 = plt.subplots()    

 

#ax1.set_title('%1.0f mK' % mean(T*1e3) )
#ax1.plot(I*1e9, V*10e3, '-')    
#
#ax1.set_xlabel('I (nA)')
#ax1.set_ylabel('V (mV)')
#
#ax2 = ax1.twinx()
#
#Rdiff = np.diff(V)/np.diff(I)
#Rdiff = np.append(Rdiff, Rdiff[-1])
#ax2.plot(I*1e9,Rdiff, 'g--', label = ' finite differencies ' )
#
#
#Rdiff =  R0byFit (I,V,n = 5)
#ax2.plot(I*1e9,Rdiff, 'k:', label = 'linear fitting ' )
#
#
#Rdiff = diffArr(I,V, 10*Istep)
#ax2.plot(I*1e9,Rdiff, 'b-.', label = 'finite difference w\ avg ' )
#
#
#Rdiff = R0byFit(I,V,10)
#Rdiff = Rdiff_TVReg(V, Istep)
#ax2.plot(I*1e9,Rdiff, 'r-', label = ' total-variation regularization ' )





    
ax2.set_yscale('log')
ax2.set_ylabel('R (Ohm)')


ax2.legend()  

    






fig,ax1 = plt.subplots()    

ax1.plot(Tarr,IcArr*1e9, 'b-x', label = 'Fitting parameter $I_1$(T), from fit')  

ax1.set_xlabel('T (mK)')
ax1.set_ylabel('$I_1$ (nA)')



ax2 = ax1.twinx()
ax2.plot(Tarr, Isw*1e9, 'r-x', label = 'Switching current Isw(T), from exp data') 
ax2.set_ylabel('Isw (nA)')
   


#ax1.set_yscale('log')
ax1.legend(loc = 1)
ax2.legend(loc = 3)  


fig,ax1 = plt.subplots()    

   
ax1.plot(Tarr[:],1*Ej*1e3/gamArr[:], 'r-x', label = '$T_{eff}$($T_{ph}$)')

ax1.set_xlabel('$T_{ph}(mK)$')
ax1.set_ylabel('$T_{eff}(mK)$')

#ax1.set_yscale('log')
ax1.legend()
  


  
  
Teff = (2*Ej/gamArr[:])



#print (a)

if Frust == 'IV ZF':    
    fig,ax1 = plt.subplots()    
 
    a, b = polyfit (1/Teff,np.log(R0arr*Teff), 1 )    
    ax1.plot(1/Teff, R0arr*Teff, 'rx', label = 'exp data $R_0 \cdot T_{eff}$')  
    ax1.plot(1/Teff, np.exp((a/Teff + b)), '-', label = 'fit to $R \cdot T \sim e^{2E_j/T}$, $E_J$ = %1.1f K' % (-a/2))  
    #   
    #ax1.plot(1/Tarr,Rmax, 'r-x', label = 'R_mmin')
    
    ax1.set_xlabel('1/$T_{eff}$ ($K^{-1}$)')
    ax1.set_ylabel('$R_0 T$ (Ohm K)')
    
    ax1.set_yscale('log')
    ax1.legend(fontsize = 13)






    fig,ax1 = plt.subplots()    
 
    ax1.plot(1e3/Tarr, R0arr, 'b-x', label = 'exp data $R_0$')  
    #   
    #ax1.plot(1/Tarr,Rmax, 'r-x', label = 'R_mmin')
    
    ax1.set_xlabel('1/$T_{phys}$ ($K$)')
    ax1.set_ylabel('$R_0 $ (Oh)')
    
    ax1.set_yscale('log')
    ax1.legend(fontsize = 13)


  
#fig,ax1 = plt.subplots()    
#ax1.plot(Tarr,Isw, 'b-x') 
#ax1.set_yscale('log')
#

if Frust == 'IV FF':  
    fig,ax1 = plt.subplots()    
    
    act, b = polyfit (1e3/Tarr [-4:] , np.log (R0arr [-4:]), 1 )
    
    ax1.plot((1e3/Tarr)[1:],R0arr[1:], 'b-x', label = "zero bias $R_0$, activation E = %1.2f K " % act) 
    
    act, b = polyfit (1e3/Tarr [-4:] , np.log (Rmax [-4:]), 1 )
    
    ax1.plot((1e3/Tarr)[1:],Rmax[1:], 'r-x', label = "maximum $R_{max}$ , activation E = %1.2f K " % act) 
    
    ax1.set_yscale('log')
    ax1.set_xlabel('1/$T_{ph}$ ($K^{-1}$)')
    ax1.set_ylabel('$R$ (Ohm)')
    ax1.legend()
    
    
    
    #
    fig,ax1 = plt.subplots()
    
    n, b = polyfit (np.log(Tarr [-6:]/1e3) , np.log(IRmax [-6:]), 1 )
    
    ax1.plot(Tarr[1:]/1e3,IRmax[1:]*1e9, 'r-x', label = 'position of $R_{max}$; $I_{R_{max}} \sim T^{%1.1f}$' %n)  
    
    ax1.set_xlabel('$T_{ph}$ ($K$)')
    ax1.set_ylabel('$I$ (nA)')
    ax1.legend(fontsize = 14)

#ax1.set_xscale('log')
#ax1.set_yscale('log')




#
#a, b = polyfit (1/Tarr [-6:] , np.log (Rmax [-6:]), 1 )
#
#print ('Rmax %1.2f' % (a/1e3) )
#
#ax1.plot(1/Tarr[-6:],np.log (Rmax [-6:]), 'gx', 1/Tarr[-6:], a*1/Tarr[-6:] + b, 'k-', label = 'R_max')
#ax1.legend()
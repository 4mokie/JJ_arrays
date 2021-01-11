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


Frust = 'IV ZF'
#spl = 'chain 68N5\\' + Frust + '\\'
spl = 'chain 678N1\\' + Frust + '\\'

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

Tphys = np.array([])
Teff1 = np.array([])
Teff2 = np.array([])
IR = np.array([])
Iceff = np.array([])
Qarr = np.array([])


IcArr =  np.array([])
gamArr =  np.array([])



figsize = (10,35)
cols = 2
gs = gridspec.GridSpec(len(os.listdir(fdir + spl)) // cols + 1, cols)
gs.update(hspace=0.5)
gs.update(wspace=.3)

fig1 = plt.figure(num=1, figsize=figsize)
ax1 = []
ax2 = []

Ej = 1.52*2/3


for i, f in enumerate(os.listdir(fdir + spl)[8:]):


    I, V = [], []
    
    data = np.genfromtxt(fdir + spl+f, skip_header = 22 ) [1:,:] 
#    print(np.mean(data[:,5]))

    T = data[:,5]
    Iraw = data[:,7]
    Vraw = data[:,8] 


    
 
    R_sl = slice(5,-5)

    if Frust == 'IV ZF':
        Istep = 10e-11
        
        ind = abs(Vraw) < (40e-6 *1)
        Iavg, Vavg = avg_group(Iraw[ind], Vraw[ind])

        I_eqst, V_eqst = XYEqSp(Iavg, Vavg, Istep)



        I, V = offsetRemove(I_eqst, V_eqst, Istep)  
        I, V = I_eqst, V_eqst  

        
        ind = abs(I) > (0)
        # V = V / 20
        I = I[ind]
        V = (V/20)[ind]

        Rdiff = R0byFit(I,V,10)[R_sl]
        Rdiff = Rdiff_TVReg(V, Istep)[R_sl]
#        Rdiff = np.diff(V)/np.diff(I)
#        Rdiff = np.append(Rdiff, Rdiff[-1])[R_sl]

        I_R = I[R_sl]
        
    else:
        Istep = 10e-11
        
        Iavg, Vavg = avg_group(Iraw, Vraw)
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

#    ax1[-1] = ax1[0]

    ax1[-1].set_title('%1.0f mK' % mean(T*1e3) )
    ax1[-1].plot(I*1e9, V*1e3, '--')    

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
    
#    ax2.plot(I_R*1e9,Rdiff, 'r', label = 'exp $R_0$ = %1.10sOhm' % eng_string((Rdiff[np.argmin(abs(I_R))])) )

##########    
    ax2.set_yscale('log')
    ax2.set_ylabel('R (Ohm)')

    

#    ax2.set_xlim (-1e-10,1e-10)
#    ax1.set_ylim (-0.7e-5,-0.2e-5)
    
#    gam = 5
    
#    Rn = 66e5
#    Ic = np.max(abs(I))   
    
    h = 6.64e-34
    e = 1.6e-19

    Ic = 80.0e-9
 #    T = 0.3
#        Ej = 0.8
    Ec = 0.2
    Rn = 6e3  

    Ej = .8
    R = Rqp(Rn,mean(0.05))
    
    Q = 30#Qp(Ej,Ec,Rn)

    Tbase = np.mean(T)
    mode = 'PD'
    V_fit = lambda I_var, Q , gam : V_PD (I_var/40e-9, Ej/gam , Ej , Ec , Q , Rn, mode   )
#    V_fit = lambda I_var, Q, T : ( 1*V_MQT (I_var/Ic, T , Ej , Ec , Q , Rn, mode   ) + 1*V_PD (I_var/Ic, T , Ej , Ec , Q , Rn, mode   ))
    V_sinh = lambda I_var, a,  alp : abs(a* np.sinh(abs(I_var) * -alp) )

#    V_fit = lambda I_var, Ic, gam : V_AH (I_var, Ic , Ej/gam  ,Ej,  Rn   )
#    V_fit = lambda I_var, Ic, gam : V_Likh (I_var, Ic , Ej/gam  ,Ej,  Rn  )



#    R_AmbHal = lambda x, Ic,gam : -V_AmbHal(x, Ic, gam)/Ic*gam*np.sinh(x/Ic) - x/Ic*V_AmbHal(x, Ic, gam)/Ic/(1 - (x/Ic)**2)**0.5 + Rn*3.14*gam*(1-(x/Ic)**2)**0.5*np.exp(-gam*((1-(x/Ic)**2)**0.5 + (x/Ic)*np.arcsin((x/Ic))))*np.cosh(0.5*3.14*gam*(x/Ic))   
    
    
#    popt1 = popt = (Q, 0.6)
    
    popt1 = popt = (Q, Ej/Tbase)
#    

    if Frust == 'IV ZF':
     for i in range(1):

        
        try:
             popt, pcov = curve_fit(V_fit, I, V, p0 = popt1 )
        except RuntimeError:
         print ('¯\_(ツ)_/¯')
         popt = popt1

#        try:
##             sll1 = slice(2,-8)
#             mid = round( len(I)/2 )
#             sll = slice(4, mid + 4)
#
#             n, b = polyfit ( I[sll] , np.log( V [sll]), 1 )
##             n1, b1 = polyfit ( I[sll1] , np.log( V [sll1]), 1 )
##             n = (n-n1)/2
##             b = (b+b1)/2
#
#
#             poptq, pcovq = curve_fit(V_sinh, I , V , p0 = ( 3e-4 ,-n) )
#
#        except RuntimeError:
#         print ('¯\_(ツ)_/¯')
##         popt = popt1
#         poptq, pcovq = popt , popt1
#        popt[1]=Ej/Tbase
 ################   

#        ax1[-1].plot(I*1e9, V_fit(I, *popt)*1e3, 'b.', label = ' $Ic$ = %1.1f nA, $T_{eff}$ = %1.2f mK' % (popt[0]*1e9, Ej/popt[1]*1e3) )

        ax1[-1].plot(I*1e9, V_fit(I, 40, Ej/Tbase)*1e3, 'b.', label = ' $Ic$ = %1.1f nA, $T_{eff}$ = %1.2f mK' % (popt[0]*1e9, Ej/popt[1]*1e3) )

        ax1[-1].set_xlim (-1.5,1.5)
        ax1[-1].set_ylim (-0.002,0.002)

#        ax1[-1].plot(I*1e9, V_fit(I, *popt)*1e3, 'b.', label = ' $Q$ = %1.1f , $T_{eff}$ = %1.2f mK, R= %1.2f' % (popt[0], popt[1]*1e3,R/1e6) )

#        ax1[-1].plot(I*1e9, V_fit(I,  *popt)*1e3, 'b.', label = ' $T_{eff_sihn}$ = %1.1f mK, $T_{effAH}$ = %1.2f ' % (1e3*Flux0/2/n/kB, Ej/popt[1]) )

#        ax1[-1].plot(I*1e9 , np.exp(n*I+b)*1e3 , label = ' $T_{eff_sihn}$ = %1.1f mK, $T_{effAH}$ = %1.2f mK ' % (-1e3*Flux0/2/n/kB,1e3* Ej/popt[1]) )
#        ax1[-1].plot(I*1e9 , V_sinh(I, poptq[0], poptq[1])*1e3, label = ' $a$ = %1.1f uV , $T eff$ = %1.2f mK' % (poptq[0]*1e6, 1e3*Flux0/2/kB/poptq[1])  )


#        ax1[-1].plot(I*1e9, V_AmbHal(I,  *popt1)*10e3, 'rx')
#        ax1[-1].plot(I*1e9, V_PD (I/Ic, T , Ej , Ec , Q, Rn )*1e3 , 'g--', label = ' V pd ' )

        ax1[-1].legend()
#        ax1[-1].set_yscale('log')
     
#        ax2.plot(I*1e9,R_AmbHal(I, *popt),'b.' )
#
#        ax2.plot(I*1e9,R_AmbHal(I, *popt1),'rx')


        ax2.set_yticks([10])
        
        Tphys = np.append(Tphys, mean(1e3*T) )



        Teff2 = np.append(Teff2, abs(1e3*Flux0/2/kB/n) )

        IR = np.append(IR, abs( poptq[0] ))

        Qarr = np.append(Qarr, abs( popt[0] ))

        Iceff = np.append(Iceff, abs( popt[0] ))

        Teff1 = np.append(Teff1, 1e3*Ej/popt[1] )



        Isw = np.append(Isw, -np.min((I)) )
#######################
#    plt.legend()
    

#  

fig,ax1 = plt.subplots()    

plt.plot(Tphys, Teff1, 'r-x')
#plt.yscale('log')
#plt.plot(Tphys, Teff1, 'r-x', label = 'Teff')
plt.plot(Tphys, Tphys, 'g--', label = 'Tphys')
ax1.set_xlabel('Tphys (mK)')
ax1.set_ylabel('Teff (mK)')

#plt.plot(Tphys, Teff2, 'b-x', label = 'Tphys')


#plt.yscale('log')

plt.legend()



fig,ax1 = plt.subplots()    

plt.plot(Tphys , Iceff*1e9,'x-', label = ' Qarr'  )
ax1.set_xlabel('Tphys (mK)')
ax1.set_ylabel('Isw (nA)')


#plt.yscale('log')

#fig,ax1 = plt.subplots()    
#
#ax1.plot(Tphys , Ic/Iceff,'rx-'  )
#ax2 = ax1.twinx()
#
#ax2.plot(Tphys, Isw)
#
#
#plt.yscale('log')




 

#ax1.set_title('%1.0f mK' % mean(T*1e3) )
#ax1.plot(I*1e9, V*1e3, '-')    
#
#ax1.set_xlabel('I (nA)')
#ax1.set_ylabel('V (mV)')
#
#ax2 = ax1.twinx()
#
#Ic = 3.0e-9
#T = 0.15
#Ej = 0.2
#Ec = 0.9
#Q = 1
#Rn = 10e0
#
#ax1.plot(I*1e9,V_PD (I/Ic, T , Ej , Ec , Q, Rn )*1e3 , 'g--', label = ' V pd ' )
#
#ax1.plot(I*1e9,V_fit (I, Ic, Ej/T )*1e3 , 'b.', label = ' Vfit' )





    







  



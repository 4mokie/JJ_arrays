# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:28:59 2018

@author: kalas
"""

import numpy as np
import os
from tvregdiff import *
from JJformulas import *

def avg_group(vA0, vB0):
    vA0 = np.round(vA0*1e15)/1e15   #remove small deferences
    vB0 = np.round(vB0*1e15)/1e15
    
    vA, ind, counts = np.unique(vA0, return_index=True, return_counts=True) # get unique values in vA0
    vB = vB0[ind]
    for dup in vB[counts>1]: # store the average (one may change as wished) of original elements in vA0 reference by the unique elements in vB
        vB[np.where(vA==dup)] = np.average(vB0[np.where(vA0==dup)])
    return vA, vB


def cut_dxdy(vA0, vB0, dx,dy):
    
    ind1 = np.where(np.abs(vA0) < dx )
    vA1, vB1 = vA0[ind1], vB0[ind1]

    ind2 = np.where(np.abs(vB1) < dy )
    vA, vB = vA1[ind2], vB1[ind2]

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


def Rdiff_TVReg(V, Istep):
    stepx = 0.05
    Rdiff = (TVRegDiff(V, 100, 10e-3, dx = stepx, ep=1e-1, scale='small', plotflag=0)*stepx / Istep)[:-1]
    return Rdiff


def XYEqSp(Xarr, Yarr, step):
    outX = []
    outY = []

    n = int((np.max(Xarr) - np.min(Xarr)) // step)    
    
    for i in range(n):
        outX = np.append( outX, V_func(Xarr, Xarr, np.min(Xarr) + i*step)  )
        outY = np.append( outY, V_func(Xarr, Yarr, np.min(Xarr) + i*step)  )

    return outX, outY


def offsetRemove(I,V, Istep, mode = 'ZF', Ioff_def = 8e-12):
       
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
   

    return Inew, Vnew


def IVC_symmer(I,V):
    
    I_off = ( np.max(I) + np.min(I) )/2
    V_off = ( np.max(V) + np.min(V) )/2
    V_off = 0
    return I - I_off, V - V_off 

# def Rdiff_TVReg(V, Istep):
#     stepx = 0.05
#     Rdiff = (TVRegDiff(V, 100, 10e-3, dx = stepx, ep=1e-1, scale='small', plotflag=0)*stepx / Istep)[:-1]
#     return Rdiff

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



def load_IVC_B(file):

    IVC = []
    
    data = np.genfromtxt(file, skip_header = 22 ) [1:,:] 
    Ts    = data[:,5]
    Iraw = data[:,7]
    Vraw = data[:,8]
    IG   = data[:,6]

    

    index_sets = [np.argwhere(i == IG) for i in np.unique(IG)]

    Iss = []
    Igs = []

    for sll in index_sets:
        sl = sll.flatten()

#         I, V = avg_group(Iraw[sl], Vraw[sl])
        I, V = Iraw[sl], Vraw[sl]
        
        T = np.mean(Ts[sl])
        B = np.mean(IG[sl])
        
        IVC_i = {'I' : I, 'V' : V, 'T' : T, 'B' : B }
        
        IVC.append(IVC_i)

    return IVC



def plot_IVC(ax, IVC, cut = False, plotRd = False):
    
    I = IVC['I']
    V = IVC['V']
    B = IVC['B']
    
    

    
    cosφ =  np.abs( np.cos(np.pi/2*B/8.85e-4))
    
    IVC['cosφ'] = cosφ
    
    if cut:
        I, V = cut_dxdy(I, V, dx = 5e-9 ,dy = 3.85e-5)
        I, V = IVC_symmer(I,V)

#         dI_max = np.max (np.diff (I) )
#         I, V =  XYEqSp(I, V, step = dI_max)
        
#         R0 = Rdiff_TVReg(V, Istep = dI_max )
#         ax.plot (I, I*np.min(  np.abs(R0) ))
    
    ax.plot(I ,V, 'o-',  label = 'cos = {:1.2f}'.format(cosφ))

    
    
    if plotRd:
        Rds = Rdiff_TVReg(V, Istep = 1e-10)
        ax2 = ax.twinx()
        ax2.plot(I, Rds)
        


# def fit_to_IZ(IVC, )













    
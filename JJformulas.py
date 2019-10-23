# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:28:59 2018

@author: kalas
"""


import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import numpy as np
import os

from tqdm import tqdm, tqdm_notebook
import math

from  scipy.special import digamma as ψ
from mpmath import besseli as Iν
from scipy.constants import hbar, pi, h, e, k

kB = k
γ = 0.57721566
RQ = h/4/e**2
Φ0 = h/2/e  
Δ = 2.1*kB

def Ic (R, Δ = 2.1*kB):
    return pi*Δ/2/e/R



def EJ_AB (R, Δ = 2.1*kB):
    
    Ic_ = Ic (R, Δ)
    
    return Ic_*Φ0/2/pi/kB


def EJ_star (EJ, R, T, C):
    β = 1/T/kB
    ωR = 1/R/C
    ρ = R/RQ
    
    return EJ*np.exp( -ρ*( ψ(1 + hbar*ωR*β/2/pi)+ γ) )


def EJ_star_simp (EJ, C):
    
    EC = e**2/C
    α = kB*EJ/EC/4
    
    return EJ*α/( 1+ α)

def  I_IZ( Vb, EJ, R, T):
    out = []

    β = 1/T/kB
    ρ = R/RQ
    Z = 1j*β*e*Vb/pi/ρ

    
    for z in Z:
       
        try :
            out = np.append(out, 2*e/hbar*EJ*kB * ( float((Iν(1-z, EJ/T) / Iν(-z, EJ/T)).imag) ))
        except OverflowError:
            print('¯\_(ツ)_/¯')
            out = np.append(out, 0 )

    return out

def find_R0_Isw( EJ, R_env , T, VERBOSE = False):
    
    
    
    Ic0 = EJ/ (Φ0/2/pi/kB)
    Vc0 = R_env*Ic0
    
    Vs = np.linspace(0, 2*Vc0, 201)
    
    Is = I_IZ( Vs, EJ = EJ, R = R_env, T = T)
    
    Is_max = np.max (Is)
    R0 = np.mean( (np.diff( Vs - 1*Is*R_env)/np.diff(Is))[:11] ) + 1 
    
    if VERBOSE:
        fig, ax = plt.subplots()
        
        ax.plot(Vs - 1*Is*R_env, Is)
        ax.axhline(Is_max, 0,1, ls = '--', label = r'$I_s = {:2.1f} nA$'.format(Is_max/1e-9))
        
        Iss = np.linspace (0, Ic0, 51)
        ax.plot( R0*Iss, Iss, label = r'$R_0 = {:2.3f} kOhm$'.format(R0/1e3) )
        
        ax.legend()
        
#         fig.close()
        print(Is_max)
        print(Ic0)
    
    return R0, Is_max

def find_Isw( RN, R_env , T, C ):

    Vs = np.linspace(0, 10e-3, 51)
    
#     EJ_s = EJ_star (EJ = EJ_AB(RN), R = R_env, T = T, C = C)
    EJ_s = EJ_star_simp (EJ = EJ_AB(RN),  C = C)
    
    
    Is = I_IZ( Vs, EJ = EJ_s, R = R_env, T = T) 

    return np.max(Is)


def find_R0( RN, R_env , T, C ):

    Vs = np.linspace(0, .1e-5, 51)
    
#     EJ_s = EJ_star (EJ = EJ_AB(RN), R = R_env, T = T, C = C)
    EJ_s = EJ_star_simp (EJ = EJ_AB(RN),  C = C)
    
    Is = I_IZ( Vs, EJ = EJ_s, R = R_env, T = T) 
    
    return np.mean(np.diff(Vs - Is*R_env)/np.diff(Is)) + 1


def  V_AH( I, Ic, T, EJ, Rn):
    vs = []
    Γ = 2*EJ/T
    i_s = I/Ic
    
    for i in i_s:
        if i < 0.95:
            i_ = (1 - i**2)**0.5
            v = 2*Ic*Rn* i_ * np.exp( -Γ*( i_ + i*np.arcsin(i) ))*np.sinh(np.pi/2*Γ*i)
        elif i > 1.05:
            v = Ic*Rn*(i**2 - 1)**0.5
        else:
            v = np.nan    
        vs.append(v)
    return np.array(vs)

def  II( R1, R2):
    return R1*R2/(R1+R2)

def  Iqp(  V, T, G1 = 1/6.76e3, G2 = 1/60e3, V0 = 0.15e-3 ):
    
    
    

    I = ( (G1-G2)*V0*np.tanh(V/V0) + G2*V)*np.exp(-Δ/T/kB)
    
    return I

#####################################################
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




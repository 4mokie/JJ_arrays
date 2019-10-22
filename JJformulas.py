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

def  Rj( RN, T ):
    α = 1e1
    return α*RN*np.exp( -1/Δ*T*kB )

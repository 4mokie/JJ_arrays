# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:28:59 2018

@author: kalas
"""


import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import numpy as np
import os
import sympy as sp

from scipy import integrate

from tqdm import tqdm, tqdm_notebook
import math

from qcodes.dataset.plotting import plot_by_id, get_data_by_id

from  scipy.special import digamma as ψ, gamma
import scipy.special as special

from mpmath import besseli as Iν
from scipy.constants import hbar, pi, h, e, k

kB = k
γ = 0.57721566
RQ = h/4/e**2
Φ0 = h/2/e  
Δ = 2.1*kB


def Qp(EJ, Ec, Rsh):
#    return wpK(EjK(Ic), EcK(C))*kB/hbar*C*Rshunt
    return np.pi*Rsh/RQ*(EJ/2/Ec)**0.5


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


def  V_AH_star( I,  EJ, Rn,  T):
    Ic0 = EJ/( Φ0/2/pi/kB )
    i = I/Ic0
    Γ = 2*EJ/T
    i_ = (1 - i**2)**0.5
    
    return 2*Ic0*Rn* i_ * np.exp( -Γ*( i_ + i*np.arcsin(i) ))*np.sinh(np.pi/2*Γ*i)
    


def  V_AH( I,  EJ, Rn,  T):
    vs = []
    
    Ic0 = EJ/( Φ0/2/pi/kB )
    
    Γ = 2*EJ/T
    i_s = I/Ic0
    
    for i in i_s:
        if i < 0.95:
            i_ = (1 - i**2)**0.5
            v = 2*Ic0*Rn* i_ * np.exp( -Γ*( i_ + i*np.arcsin(i) ))*np.sinh(np.pi/2*Γ*i)
        elif i > 1.05:
            v = Ic0*Rn*(i**2 - 1)**0.5
        else:
            v = np.nan    
        vs.append(v)
    return np.array(vs)

def  II( R1, R2):
    return R1*R2/(R1+R2)

def  Iqp(  V, T, G1 = 1/6.76e3, G2 = 1/60e3, V0 = 0.15e-3 ):
    I = ( (G1-G2)*V0*np.tanh(V/V0) + G2*V)*np.exp(-Δ/T/kB)
    
    return I


def R0_IZ(EJ, R, T):
    
    return R/(Iν(0, EJ/T)**2 - 1 )


def Njump(i, Q, EJ, T):
    
    Γ = T/EJ
    
    z = (8/np.pi/Q/Γ)**0.5
    Ee = 8/np.pi**0.5/Q*( np.exp(-z**2)/z/special.erfc(z) - np.pi**0.5 ) 
    
    Ed = (1 + np.pi**2/8*Ee)**0.5
    im = np.pi/4*i*Q
    
    Np = 1 + 2*Q/np.pi**2*( Ed - 1 ) + i*Q**2/2/np.pi*np.log( ( Ed - im )/(1 - im) )
    Nm = 1 + 2*Q/np.pi**2*( Ed - 1 ) - i*Q**2/2/np.pi*np.log( ( Ed + im )/(1 + im) )

    
    return Np, Nm


def wpK(EjK, EcK):
    return np.sqrt(8*EjK*EcK)

def ΔU(i, EJ):
    ΔUp = 2*EJ*( 1*(1-i**2)**0.5 + i*(np.arcsin(i) - np.pi/2) )
    ΔUm = 2*EJ*( 1*(1-i**2)**0.5 + i*(np.arcsin(i) + np.pi/2) )

    return ΔUp, ΔUm

def τ(i, EJ, Ec, T):
    
    ωa = wpK(EJ, Ec)*kB/hbar * (1 - i**2)**0.25
    
    ΔUp, ΔUm = ΔU(i, EJ)
    
    τp = 2*np.pi/ωa*np.exp( ΔUp/T )
    τm = 2*np.pi/ωa*np.exp( ΔUm/T )
    
    return τp, τm

def τQ(i, EJ, Ec, T):
    ωa = wpK(EJ, Ec)*kB/hbar * (1 - i**2)**0.25

    ΔUp, ΔUm = ΔU(i, EJ)
        
    aqp = ( 864*np.pi*ΔUp/wpK(EJ,Ec) )**0.5
    aqm = ( 864*np.pi*ΔUm/wpK(EJ,Ec) )**0.5
    
    τQp = 2*np.pi/ωa /aqp  *np.exp(7.2* ΔUp / wpK(EJ,Ec) ) 
    τQm = 2*np.pi/ωa /aqm  *np.exp(7.2* ΔUm / wpK(EJ,Ec) )
    return  τQp, τQm


def V_KM(I, EJ, Ec, Q, T):
    
#     out = [np.nan for i in I]
    
    Ic0 = EJ/( Φ0/2/pi/kB )
    
    i = I/Ic0
    
    τp, τm = τ(i, EJ, Ec, T)
    Np, Nm = Njump(i, Q, EJ, T)
    
    τQp, τQm = τQ(i, EJ, Ec, T)
    
    out = h/2/e*(Np/τp +1/τQp - Nm/τm - 1/τQm)
    
    out[np.where (abs(i) > 4/np.pi/Q) ] = np.nan 
    
    return out
    

def R0_KM(EJ, Ec, Q, T):    
    di = 0.01
    Ic0 = EJ/(Φ0/2/pi/kB)
    R0 = V_KM(di, EJ, Ec, Q, T)/di/Ic0
    
    return R0
# #####################################################
# def avg_group(vA0, vB0):
#     vA0 = np.round(vA0*1e15)/1e15   #remove small deferences
#     vB0 = np.round(vB0*1e15)/1e15
    
#     vA, ind, counts = np.unique(vA0, return_index=True, return_counts=True) # get unique values in vA0
#     vB = vB0[ind]
#     for dup in vB[counts>1]: # store the average (one may change as wished) of original elements in vA0 reference by the unique elements in vB
#         vB[np.where(vA==dup)] = np.average(vB0[np.where(vA0==dup)])
#     return vA, vB


# def cut_dxdy(vA0, vB0, dx,dy):
    
#     ind1 = np.where(np.abs(vA0) < dx )
#     vA1, vB1 = vA0[ind1], vB0[ind1]

#     ind2 = np.where(np.abs(vB1) < dy )
#     vA, vB = vA1[ind2], vB1[ind2]

#     return vA, vB


# def V_func(I,V, val):
#     out = []
#     for x in np.nditer(val):
#         out = np.append (out,  V[np.argmin(abs(I-x))])
#     return out


# def diffArr(Xarr, Yarr, step):
#     out = []
#     for x in Xarr:
#         out = np.append(out, np.mean((V_func(Xarr,Yarr, x+step/2))  - np.mean(V_func(Xarr,Yarr, x-step/2)))/(step))
#     return out


# def R0byFit (I,V,n = 3):
#     V = np.append(V, V[-n:])
#     I = np.append(I, I[-n:])
    
#     out = []
    
#     for i in range(len(I)-n):    
#         a, b = polyfit (I [i:i+n] , V [i:i+n], 1 )
#         out = np.append(out, a)
        
#     return out

# def XYEqSp(Xarr, Yarr, step):
#     outX = []
#     outY = []

#     n = int((np.max(Xarr) - np.min(Xarr)) // step)    
    
#     for i in range(n):
#         outX = np.append( outX, V_func(Xarr, Xarr, np.min(Xarr) + i*step)  )
#         outY = np.append( outY, V_func(Xarr, Yarr, np.min(Xarr) + i*step)  )

#     return outX, outY


# def offsetRemove(I,V, Istep, mode = 'ZF', Ioff_def = 8e-12):
       
#     Rdiff = Rdiff_TVReg(V, Istep)
    
#     ind_minR = np.argmin(Rdiff)
#     ind_maxR = np.argmax(Rdiff)


#     if mode == 'ZF':
#         Ioff = I[ind_minR]
#     else:
#         Ioff = Ioff_def
  
#     Voff = V_func(I, V, Ioff)

#     Inew = I - Ioff
#     Vnew = V - Voff
   

#     return Inew, Vnew

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


#Tom's additions:

def Eb(I):
    return I*Φ0/(2*pi*k)

def Ij(Ej):
    return Ej*2*pi*k/Φ0

def I_qsm(V,Renv,T,Ej,Ec):
    rho = Renv/RQ
    #beta = 1/(kB*T)
    beta = 1/T
    Ejstar = Ej*rho**rho*(beta*Ec/(2*pi**2))**(-1*rho)*np.exp(-1*rho*γ)
    Iqsm = (e*rho*beta*pi/hbar)*(Ejstar)**2*(beta*e*V/kB)/((beta*e*V/kB)**2+pi**2*rho**2)
    
    return Iqsm


def I_cb_ig(V,Renv,T,Ej,Ec):
    rho = Renv/RQ
    #beta = 1/(kB*T)    
    beta = 1/T
    L = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*(pi**2)*rho)))
    Icb = beta*Ej*np.exp(-L)/(4*pi)*np.abs(special.gamma(rho - 1j*beta*e*V/kB/pi))**2/special.gamma(2*rho)*np.sinh(beta*e*V/kB)
    
    return Icb

def I_cb_ig_gammainput(V,Renv,T,Ej,L,G):
    rho = Renv/RQ
    #beta = 1/(kB*T)
    beta = 1/T
    Icb = beta*Ej*np.exp(-L)/(4*pi)*G*np.sinh(beta*e*V/kB)
    
    return Icb

def V_qsm(Renv,T,Eb,Tqm):
    rho = Renv/RQ
    #beta = 1/(kb*T)
    beta = 1/T
    #Eb = Ib*Φ0/(2*pi)/kB #check all the kB's!!!!!
    Vqsm = rho*pi/(beta*e)*(1-np.exp(-2*pi*beta*Eb))/Tqm
    #print('pi*beta*Eb = ' + str(pi*beta*Eb))
    
    return Vqsm

def V_qsm_full_verbose(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*(pi**2)*rho)))
    theta = Lambda*beta*Ej
    Ejstar = Ej*(1-Lambda/2)
    Tintegral = integrate.dblquad(lambda phi,phip: np.exp(-beta*Eb*phi)*np.exp(beta*Ejstar*np.cos(phip)*np.sin(phi/2))*(1-theta*np.sin(phip-phi/2))*np.exp(2*beta*theta*np.sin(phip)*np.sin(phi/2)*(Eb + Ejstar*np.cos(phip)*np.cos(phi/2))), 0, 2*pi, lambda phip: 0, lambda phip: 2*pi)
    Tqm = Tintegral[0]
    #rho = Renv/RQ
    #beta = 1/T
    Vqsm = rho*pi/(beta*e)*(1-np.exp(-2*pi*beta*Eb))/Tqm
    #starting here is annoying stuff to remove
    print('Renv = ' + str(Renv))
    print('T = ' + str(T))
    print('Eb = ' + str(Eb))
    print('Ej = ' + str(Ej))
    print('Ec = ' + str(Ec))
    print('beta = ' + str(beta))
    print('rho = ' + str(rho))
    print('Lambda = ' + str(Lambda))
    print('theta = ' + str(theta))
    print('Ejstar = ' + str(Ejstar))
    print('Tqm = ' + str(Tqm))
    print('Vqsm = ' + str(Vqsm))
        
    return Vqsm


def V_qsm_full(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    theta = Lambda*beta*Ej
    Ejstar = Ej*(1-Lambda/2)
    Tintegral = integrate.dblquad(lambda phi,phip: np.exp(-beta*Eb*phi)*np.exp(-2*beta*Ejstar*np.cos(phip)*np.sin(phi/2))*(1-theta*np.sin(phip-phi/2))*np.exp(2*beta*theta*np.sin(phip)*np.sin(phi/2)*(Eb + Ejstar*np.cos(phip)*np.cos(phi/2))), 0, 2*pi, lambda phip: 0, lambda phip: 2*pi)
    Tqm = Tintegral[0]
    #rho = Renv/RQ
    #beta = 1/T
    Vqsm = rho*pi/(beta*kB*e)*(1-np.exp(-2*pi*beta*Eb))/Tqm
        
    return Vqsm
#add Tqm to return

def V_qsm_scaled(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    theta = Lambda*beta*Ej
    Ejstar = Ej*(1-Lambda/2)
    Tintegral = integrate.dblquad(lambda phi,phip: np.exp(-beta*Eb*phi)*np.exp(-2*beta*Ejstar*np.cos(phip)*np.sin(phi/2))*(1-theta*np.sin(phip-phi/2))*np.exp(2*beta*theta*np.sin(phip)*np.sin(phi/2)*(Eb + Ejstar*np.cos(phip)*np.cos(phi/2))), 0, 2*pi, lambda phip: 0, lambda phip: 2*pi)
    Tqm = Tintegral[0]
    #rho = Renv/RQ
    #beta = 1/T
    Vqsm = 10**6*rho*pi/(beta*e)*(1-np.exp(-2*pi*beta*Eb))/Tqm
        
    return Vqsm



def V_qsm_full_test2(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    theta = Lambda*beta*Ej
    S = γ + pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho))
    Ejstar = Ej*np.exp(-rho*S)
    #Ejstar = Ej*(1-Lambda/2)
    Tintegral = integrate.dblquad(lambda phi,phip: np.exp(-beta*Eb*phi)*np.exp(-2*beta*Ejstar*np.cos(phip)*np.sin(phi/2))*(1-theta*np.sin(phip-phi/2))*np.exp(2*beta*theta*np.sin(phip)*np.sin(phi/2)*(Eb + Ejstar*np.cos(phip)*np.cos(phi/2))), 0, 2*pi, lambda phip: 0, lambda phip: 2*pi)
    Tqm = Tintegral[0]
    #rho = Renv/RQ
    #beta = 1/T
    Vqsm = rho*pi/(beta*e)*(1-np.exp(-2*pi*beta*Eb))/Tqm
        
    return Vqsm
    

def V_qsm_full_test3(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    theta = Lambda*beta*Ej
    #S = γ + pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho))
    Ejstar = Ej*rho**rho*(beta*Ec/(2*pi**2))**(-1*rho)*np.exp(-rho*γ)
    #Ejstar = Ej*(1-Lambda/2)
    Tintegral = integrate.dblquad(lambda phi,phip: np.exp(-beta*Eb*phi)*np.exp(-2*beta*Ejstar*np.cos(phip)*np.sin(phi/2))*(1-theta*np.sin(phip-phi/2))*np.exp(2*beta*theta*np.sin(phip)*np.sin(phi/2)*(Eb + Ejstar*np.cos(phip)*np.cos(phi/2))), 0, 2*pi, lambda phip: 0, lambda phip: 2*pi)
    Tqm = Tintegral[0]
    #rho = Renv/RQ
    #beta = 1/T
    Vqsm = rho*pi/(beta*e)*(1-np.exp(-2*pi*beta*Eb))/Tqm
        
    return Vqsm
    
    

def V_cl(Renv,T,Eb,Tcl):
    rho = Renv/RQ
    #beta = 1/(kb*T)
    beta = 1/T
    #Eb = Ib*Φ0/(2*pi)/kB #check all the kB's!!!!!
    Vcl = rho*pi/(beta*e/kB)*(1-np.exp(-2*pi*beta*Eb))/Tcl
    #print('pi*beta*Eb = ' + str(pi*beta*Eb))
    
    return Vcl


def V_cl_full(Renv,T,Eb,Ej):
    rho = Renv/RQ
    #beta = 1/(kb*T)
    beta = 1/T
    Tintegral = integrate.quad(lambda phi: np.exp(-beta*Eb*phi)*Iν(0,2*beta*Ej*np.sin(phi/2)), 0, 2*pi)
    Tcl = Tintegral[0]
    #Eb = Ib*Φ0/(2*pi)/kB #check all the kB's!!!!!
    Vcl = rho*pi/(beta*e)*(1-np.exp(-2*pi*beta*Eb))/Tcl
    #print('pi*beta*Eb = ' + str(pi*beta*Eb))
    
    return Vcl
#check above function...    


# Try Tcl (should give IZ theory), CB, other stuff...in PE or w/e we have Tqm = Tcl(Ej*)

def T_qm(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    theta = Lambda*beta*Ej
    Ejstar = Ej*(1-Lambda/2) 
    Tintegral = integrate.dblquad(lambda phi,phip: (1/(2*pi))*np.exp(-beta*Eb*phi)*np.exp(-2*beta*Ejstar*np.cos(phip)*np.sin(phi/2))*(1-theta*np.sin(phip-phi/2))*np.exp(2*beta*theta*np.sin(phip)*np.sin(phi/2)*(Eb + Ejstar*np.cos(phip)*np.cos(phi/2))), 0, 2*pi, lambda phip: 0, lambda phip: 2*pi)
    Tqm = Tintegral
    
    #print(beta, rho, Lambda, theta, Ejstar)
        
    return Tqm


def T_qm_test(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    theta = Lambda*beta*Ej
    Ejstar = Ej*(1-Lambda/2) 
    Tintegral = integrate.dblquad(lambda phi,phip: (1/(2*pi))*np.exp(-phi)*np.exp(-2*np.cos(phip)*np.sin(phi/2))*(1-np.sin(phip-phi/2))*np.exp(2*np.sin(phip)*np.sin(phi/2)*(1 + 1*np.cos(phip)*np.cos(phi/2))), 0, 2*pi, lambda phip: 0, lambda phip: 2*pi)
    Tqm = Tintegral
    
    #print(beta, rho, Lambda, theta, Ejstar)
        
    return Tqm


def T_qm_test2(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    theta = Lambda*beta*Ej
    #Ejstar = Ej*(1-Lambda/2) 
    S = γ + pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho))
    Ejstar = Ej*np.exp(-rho*S)
    Tintegral = integrate.dblquad(lambda phi,phip: (1/(2*pi))*np.exp(-beta*Eb*phi)*np.exp(-2*beta*Ejstar*np.cos(phip)*np.sin(phi/2))*(1-theta*np.sin(phip-phi/2))*np.exp(2*beta*theta*np.sin(phip)*np.sin(phi/2)*(Eb + Ejstar*np.cos(phip)*np.cos(phi/2))), 0, 2*pi, lambda phip: 0, lambda phip: 2*pi)
    Tqm = Tintegral
    
    #print(beta, rho, Lambda, theta, Ejstar)
        
    return Tqm


#note: make this stuff more readable, maybe with kwargs. Maybe have a function that takes Ej, Ec, etc as inputs and outputs a useful array that can be used as input for these functions? anyway, do that later.

def T_cl(T,Eb,Ej):
    #beta = 1/(kB*T)
    beta = 1/T
    #Eb = 1
    #Ej = 1
    #Tcl = 1
    
    #above is a placeholder
    Tintegral = integrate.quad(lambda phi: np.exp(-beta*Eb*phi)*Iν(0,2*beta*Ej*np.sin(phi/2)), 0, 2*pi)
    Tcl = Tintegral
    
    return Tcl 


def V_qsm1(Rj,Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Ic = Ej/(Φ0/(2*pi*k))
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    theta = Lambda*beta*Ej
    alpha = Eb/Ej
    
    Vqsm1 = Rj*Ic*np.sqrt(1-alpha**2)/(2*pi)*np.exp(-2*beta*Ej*(1-alpha**2)**(3/2)/(3*alpha**2))*np.exp(2*theta*np.sqrt(1-alpha**2))
    
    return Vqsm1
    

def log_V_qsm1(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Ic = Ej/(Φ0/(2*pi*k))
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    theta = Lambda*beta*Ej
    alpha = Eb/Ej
    
    logVqsm1 = np.log(np.sqrt(1-alpha**2)/(2*pi))+(-2*beta*Ej*(1-alpha**2)**(3/2)/(3*alpha**2))+(2*theta*np.sqrt(1-alpha**2))
                      
    return logVqsm1

def qsm_params(Renv,T,Eb,Ej,Ec):
    beta = 1/T
    rho = Renv/RQ
    Ic = Ej/(Φ0/(2*pi*k))
    Lambda = 2*rho *(γ + 2*pi**2*rho/(beta*Ec) + ψ(beta*Ec/(2*pi**2*rho)))
    Ejstar1 = Ej*(1-Lambda/2)
    theta = Lambda*beta*Ej
    Dmin = (1-theta)**(-1)
    Dmax = (1+theta)**(-1)
    alpha = Eb/Ej
    
#     print('beta = ' + str(beta))
#     print('rho = ' + str(rho))
#     print('Ic = ' + str(Ic))
#     print('digamma = ' + str(ψ(beta*Ec/(2*pi**2*rho))))
#     print('Lambda = ' + str(Lambda))
#     print('Ejstar1 = ' + str(Ejstar1))
#     print('theta = ' + str(theta))
#     print('D(cos = -1) = ' + str(Dmin))
#     print('D(cos = +1) = ' + str(Dmax))
#     print('alpha = ' + str(alpha))
    
    return Ic,Lambda,theta,Eb,alpha,2*beta*Ej*(1-alpha**2),Dmin,Dmax


def V_smol(Renv,T,Eb,Ej,Ec):
    #ref: Tafuri pg 258 eqn 7.50
    delta = 1.764*k*1.2    # gap = 1.764*kTc
    Ib = Eb*2*pi*k/Φ0
    Ij = Ej*2*pi*k/Φ0
    Rn = pi*delta/(2*e*Ij)    #pi*delta/2eIc
    gam = hbar*Ij/(e*k*T)
    C = e**2/(2*Ec)
    wj = np.sqrt(2*e*Ij/(hbar*C))
    alpha = Ib/Ij
    Omega = Rn*C*wj
    T1 = integrate.quad(lambda phi: np.exp(-gam*alpha*phi/2)*Iν(0,gam*np.sin(phi/2)), 0, 2*pi)
    T2 = integrate.quad(lambda phi1: np.exp(-gam*alpha*phi1/2)*Iν(1,gam*np.sin(phi1/2))*np.sin(phi1/2), 0, 2*pi)
    Vsmol = (2/gam)*Rn*Ij*(np.exp(pi*gam*alpha)-1)/np.exp(pi*gam*alpha)*(1/T1[0])*(1+(Omega**2)*(T2[0]/T1[0]))
    #added k factor to Vsmol b/c Rn*Ij seem to need it but then took it out
    
    return Vsmol

def V_smol_test(Renv,T,Eb,Ej,Ec):
    #ref: Tafuri pg 258 eqn 7.50
    delta = 1.764*k*1.2    # gap = 1.764*kTc
    Ib = Eb*2*pi*k/Φ0
    Ij = Ej*2*pi*k/Φ0
    Rn = pi*delta/(2*e*Ij)    #pi*delta/2eIc
    gam = hbar*Ij/(e*k*T)
    C = e**2/(2*Ec)
    wj = np.sqrt(2*e*Ij/(hbar*C))
    alpha = Ib/Ij
    Omega = Rn*C*wj
    T1 = integrate.quad(lambda phi: np.exp(-gam*alpha*phi/2)*Iν(0,gam*np.sin(phi/2)), 0, 2*pi)
    T2 = integrate.quad(lambda phi1: np.exp(-gam*alpha*phi1/2)*Iν(1,gam*np.sin(phi1/2))*np.sin(phi1/2), 0, 2*pi)
    Vsmol = (2/gam)*Rn*Ij*(np.exp(pi*gam*alpha)-1)/np.exp(pi*gam*alpha)*(1/T1[0])*(1+(Omega**2)*(T2[0]/T1[0]))
    #added k factor to Vsmol b/c Rn*Ij seem to need it but then took it out
    
    return Vsmol,T1


#This J: I think it's the correlation function from PE theory but what is the special case if any?
def J(t,Zt,T):
    beta = 1/T
    Z = Zt
    Jt = 2*np.quad(lambda w: (1/w*RQ)*np.real(Z)*(np.coth(beta*hbar*w/2)*(cos(w*t)-1)-1j*np.sin(w*t)), 0, np.inf)
    return Jt

# def PE(E):
#     P = 1/(2*pi*hbar)*np.quad(lambda t: np.exp(J(t,Zt,T) + 1j*E*t

#Formulas from Zazunov, Didier, and Hekking EPL 83, 47012 (2008)
def V_NNA_smallg(Renv, T, Ib, u, Ej, Ec):
    g = RQ/Renv
    Vc = pi*U0(Ej,Ec)/e*kB #check kB factor, used to be /kB
    V = Vc*u*np.abs(gamma(g + 1j*hbar*(1/(kB*T))*Ib/(2*e)))**2/gamma(2*g)*np.sinh(pi*hbar*(1/(kB*T))*Ib/(2*e))
    return V

def V_NNA_smallg_Ejfit(Renv,T,Ib,Ejs,Ec,cosine):
    g = RQ/Renv
    Beta = 1/kB/T
    Ej = Ejs*cosine
    Vc = pi*U0(Ej,Ec)/e/kB #check kB factor, used to be /kB
    u_calc = Beta*U0(Ej,Ec)/4/pi*(Beta*hbar*wc_paper(Renv,Ej,Ec)*np.exp(γ)/2/pi)**(-2*g)
    V = Vc*u_calc*np.abs(gamma(g + 1j*hbar*Beta*Ib/2/e))**2/gamma(2*g)*np.sinh(pi*hbar*Beta*Ib/2/e)
    return V

def V_NNA_smallg_nogamma(Renv, T, Ib, u, Ej, Ec):
    g = RQ/Renv
    Vc = pi*U0(Ej,Ec)/e/kB
    V = Vc*u*np.sinh(pi*hbar*(1/(kB*T))*Ib/(2*e))
    return V

def V_NNA_smallg_sech(Renv, T, Ib, u, Ej, Ec):
    Beta = 1/(kB*T)
    g = RQ/Renv
    Vc = pi*U0(Ej,Ec)/e/kB
    #u = Beta*U0(Ej,Ec)/4/pi*(Beta*hbar*wc*np.exp(
    #u = u(Renv,T,Ej,Ec,wc)
    V = Vc*u/np.cosh(pi*hbar*Beta*Ib/2/e)/gamma(2*g)*np.sinh(pi*hbar*Beta*Ib/(2*e))
    return V
#this assumes g = 1/2, be careful with Renv!

def V_NNA_smallg_10workaround(Renv, T, Ib, u, Ej, Ec):
    g = RQ/Renv
    Vc = pi*U0(Ej,Ec)/e/kB
    V = Vc*u*np.abs(gamma(g + 1j*hbar*(1/(kB*T))*Ib/10/(2*e)))**2/gamma(2*g)*np.sinh(pi*hbar*(1/(kB*T))*Ib/(2*e))
    return V

def V_NNA_smallg_relative(Renv, T, Ib, u, Ej, Ec):
    g = RQ/Renv
    #Vc = pi*U0(Ej,Ec)/e/kB
    Vrel = u*np.abs(gamma(g + 1j*hbar*(1/(kB*T))*Ib/(2*e)))**2/gamma(2*g)*np.sinh(pi*hbar*(1/(kB*T))*Ib/(2*e))
    return Vrel

def V_NNA_smallg_wc(Renv, T, Ib, wc, Ej, Ec):
    g = RQ/Renv
    Vc = pi*U0(Ej,Ec)/e/kB
    V = (1/kB/T)*U0(Ej,Ec)/(4*pi)*((1/kB/T)*hbar*wc*np.exp(γ)/(2*pi))**(-2*g)*Vc*np.abs(gamma(g + 1j*hbar*(1/(kB*T))*Ib/(2*e)))**2/gamma(2*g)*np.sinh(pi*hbar*(1/(kB*T))*Ib/(2*e))
    return V

def Vc(Ej,Ec):
    return pi*U0(Ej,Ec)/e/kB

def gammapart(Renv, T, Ib):
    g = RQ/Renv
    return np.abs(gamma(g + 1j*hbar*(1/(kB*T))*Ib/(2*e)))**2/gamma(2*g)
    
def U0(Ej,Ec):
    return 4*np.sqrt(2/pi)*Ec*(2*Ej/Ec)**(3/4)*np.exp(-4*np.sqrt(2*Ej/Ec))
    #this is in Kelvin, multiply by kB to get Joules

def wg(Ej,Ec): 
    return np.sqrt(2*Ej*Ec)/hbar*kB

def wc_from_u(Renv, T, uval, Ej, Ec):
    g = RQ/Renv
    return 8*(pi*kB*T)**2/(hbar*U0(Ej,Ec)*np.exp(γ))*(uval)**(-1/(2*g)) #kB questionable, this whole function may be garbage

def u(Renv,T,Ej,Ec,wc):
    g = RQ/Renv
    return (1/kB/T)*U0(Ej,Ec)/(4*pi)*((1/kB/T)*hbar*wc*np.exp(γ)/(2*pi))**(-2*g)

def Leff(Renv,Ej,Ec):
    g = RQ/Renv
    Lj = (Φ0/2/pi)**2/Ej/kB #the kB's here should allow direct comparison to R
    return 2*pi/g/np.sqrt(2*Ec/Ej)*Lj

def wc_paper(Renv,Ej,Ec):
    return Renv/Leff(Renv,Ej,Ec)

def databyid(run_id: int, **kwargs):
    from qcodes.dataset.data_set import load_by_id

    dataset = load_by_id(run_id)
    title = f"#{run_id}, Exp {dataset.exp_name} ({dataset.sample_name})"
    alldata = get_data_by_id(run_id)
    
    return ({'title':title, 'alldata':alldata})
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:28:59 2018

@author: kalas
"""

from pylab import *
from sympy import Symbol, re
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import csv
import os
import math
import mpmath

e = 1.6e-19
h = 6.64e-34
hbar = h/2/np.pi
kB = 1.38e-23
Flux0 = 2.07e-15 
Rq = 6.5e3



def IcAB( R, Delta = 2.1*kB):
    return np.pi*Delta/2/e/R

def EjK (Ic):
    return Ic*Flux0/2/np.pi/kB

def EcK(C):
    return e**2/2/C/kB

def ELK(L):
    return Flux0**2/2/L/kB


def wpK(EjK, EcK):
    return np.sqrt(8*EjK*EcK)

def bloch_band(EjK, EcK):
    return (EjK**3*EcK)**0.25*np.exp(-np.sqrt(8*EjK/EcK))


def wcK(EjK, Rn):
    return (2*np.pi/Flux0)**2 * EjK*hbar * Rn


def Qp(Ej, Ec, Rshunt):
#    return wpK(EjK(Ic), EcK(C))*kB/hbar*C*Rshunt
    return np.pi*Rshunt/Rq*(Ej/2/Ec)**0.5

def Rqp(Rn,T, Delta = 2.05):
    Rsh = 1e8
    return   1/(1/Rsh + 1/Rn/np.exp(Delta/T))

def pDeltaU(Ej, i):
    return 2*Ej*( 1*(1-i**2)**0.5 + i*(np.arcsin(i) - np.pi/2) )

def mDeltaU(Ej, i):
    return 2*Ej*( 1*(1-i**2)**0.5 + i*(np.arcsin(i) + np.pi/2) )



#def rateTA( i, T, Ej, Ec, Q ):
#    at = 4 / ( (1 + Q * T/1.8/deltaU(Ej, i) )**0.5 + 1 )**2
#    wa = wpK(Ej, Ec)*kB/hbar * (1 - i**2)**0.25
#    return at*wa/2/np.pi*np.exp( - deltaU(Ej, i) / T )
#
#
#def rateMQT( i, T, Ej, Ec, Q ):
#    aq = (120 * np.pi * 7.2 * deltaU(Ej, i) / wpK(Ej, Ec) )
#    wa = wpK(Ej, Ec)*kB/hbar * (1 - i**2)**0.25
#    return aq*wa/2/np.pi*np.exp( -7.2 * deltaU(Ej, i) / wpK(Ej, Ec) * (1 + 0.87/Q) )

def TauTA( i, T, Ej, Ec, Q, DeltaU, Rn, mode  ):
    out = []
    
    if mode == 'PD':
        wa = wpK(Ej, Ec)*kB/hbar * (1 - i**2)**0.25/(1 - i**2)**0.25
    elif mode == 'AH':
        wa = wcK(Ej, Rn)*kB/hbar * (1 - i**2)**0.5/(1 - i**2)**0.5

#    wa = 1 * (1 - i**2)**0.25
    
#    Q = Qp(Ej,Ec,Rn)
#    
#    mu = Q**2*(1 - i**2)**0.5
#    
#    norm = ((Flux0/2/np.pi)**2/(Ej*kB)/Rn)
    
    
    for j, ix  in enumerate (i): 
        
        if True:#T < 2*np.pi * DeltaU[j] / mu[j]**0.5:
#            out = np.append(out, norm*4*np.pi * Q**2 / ((4*mu[j]+1)**0.5 -1) *np.exp( DeltaU[j] / T ))
            at = 4/( (1 + Q*T/1.8/DeltaU[j])**0.5 + 1)**2
            out = np.append(out, 2*np.pi/wa[j]/at *np.exp( DeltaU[j] / T ))
#            print('<')
        
        
        else:
            out  = np.append(out, norm*Q**2 * T / DeltaU[j] *np.exp( DeltaU[j] / T ))
#            print('>')
    return out

def TauQ( i, T, Ej, Ec, Q, DeltaU, Rn, mode  ):
    out = []
    
    if mode == 'PD':
        wa = wpK(Ej, Ec)*kB/hbar * (1 - i**2)**0.25/(1 - i**2)**0.25
    elif mode == 'AH':
        wa = wcK(Ej, Rn)*kB/hbar * (1 - i**2)**0.5/(1 - i**2)**0.5

    
    for j, ix  in enumerate (i): 
        aq = ( 864*np.pi*DeltaU[j]/wpK(Ej,Ec) )**0.5
       
        out = np.append(out, 2*np.pi/wa[j] /aq  *np.exp(7.2* DeltaU[j] / wpK(Ej,Ec)*(1 + 0.87/Q) ) )
 
    return out


def pNjump(i, Q, Ej, T):
    G = T/Ej
    
    z = (8/np.pi/Q/G)**0.5
    Ee = 8/np.pi**0.5/Q*( np.exp(-z**2)/z/special.erfc(z) - np.pi**0.5 ) 
    
    Ed = (1 + np.pi**2/8*Ee)**0.5
    im = np.pi/4*i*Q
    
    return 1 + 2*Q/np.pi**2*( Ed - 1 ) + i*Q**2/2/np.pi*np.log( ( Ed - im )/(1 - im) )
    
def mNjump(i, Q, Ej, T):
    G = T/Ej
    
    z = (8/np.pi/Q/G)**0.5
    Ee = 8/np.pi**0.5/Q*( np.exp(-z**2)/z/special.erfc(z) - np.pi**0.5 ) 
    
    Ed = (1 + np.pi**2/8*Ee)**0.5
    im = np.pi/4*i*Q
    
    return 1 + 2*Q/np.pi**2*( Ed - 1 ) - i*Q**2/2/np.pi*np.log( ( Ed + im )/(1 + im) )




def V_PD (i, T, Ej, Ec, Q, Rn, mode = 'PD'):

#    pN = 1
#    mN = 1
    
    pN = pNjump(i, Q, Ej, T)
    mN = mNjump(i, Q, Ej, T)


    return Flux0*( pN/TauTA( i, T, Ej, Ec, Q, pDeltaU(Ej, i), Rn, mode ) - mN/TauTA( i, T, Ej, Ec, Q , mDeltaU(Ej, i), Rn , mode)  )

def V_MQT (i, T, Ej, Ec, Q, Rn, mode = 'PD'):

    pN = 1
    mN = 1
    
#    pN = pNjump(i, Q, Ej, T)
#    mN = mNjump(i, Q, Ej, T)


    return Flux0*( pN/TauQ( i, T, Ej, Ec, Q, pDeltaU(Ej, i), Rn, mode ) - mN/TauQ( i, T, Ej, Ec, Q , mDeltaU(Ej, i), Rn , mode)  )



def  V_AH( I,Ic, T,Ej, Rn):
    gam = 2*Ej/T
    i = I/Ic
    return Ic*Rn*2*(1-0*i**2)**0.5*np.exp(-gam*( (1-i**2)**0.5 + i*np.arcsin(i) ))*np.sinh(np.pi/2*gam*(i))
#    return Ic*Rn*np.sinh(np.pi/2*gam*(i))/1e8



def  R_AH ( x, Ic,gam) :
    return -V_AmbHal(x, Ic, gam)/Ic*gam*np.sinh(x/Ic) - x/Ic*V_AmbHal(x, Ic, gam)/Ic/(1 - (x/Ic)**2)**0.5 + Rn*3.14*gam*(1-(x/Ic)**2)**0.5*np.exp(-gam*((1-(x/Ic)**2)**0.5 + (x/Ic)*np.arcsin((x/Ic))))*np.cosh(0.5*3.14*gam*(x/Ic))   
 

def  V_Likh( I,Ic, T,Ej, Rn):
    out = []
    gam = Ej/T
    i = I/Ic
    for x in i:
       
        try :
            out = np.append(out, Ic*Rn/np.pi/gam/ (float(mpmath.besseli(1j*x*gam, gam).real))**2 *np.sinh(np.pi*gam*x) )
        except OverflowError:
            print('¯\_(ツ)_/¯')
            out = np.append(out, 0 )

    return out


def  I_IZ( Vb, Ic, T,Ej, Rn):
    out = []
    gam = Ej/T
    i = I/Ic
    Z = 2j/T*e*Vb/hbar/Rn
    
    for z in Z:
       
        try :
            out = np.append(out, Ic * ( float((mpmath.besseli(1-z, gam) / mpmath.besseli(-z, gam)).imag) ))
        except OverflowError:
            print('¯\_(ツ)_/¯')
            out = np.append(out, 0 )

    return out


    
if __name__ == "__main__":
        
    T = 0.55

#    Ic = 3e-9
#    Ej = EjK(Ic)

    Ej = 2
    Ic = kB*Ej*2*np.pi/Flux0
    
    Ec = 0.5
    Q = 20
    Rn = 60e3


    
    im = 1/(np.pi/4*1*Q)
    
    i = np.arange(0.01, im, im / 101)   
    v = np.arange(0, 0.2, 0.001)
    
    vfit = abs(V_PD (i, T , Ej , Ec , Q, Rn ))
    
    mid = round( len(i)/2 )
    sll = slice(mid - 4, mid + 4)



    n, b = polyfit ( i[sll] , np.log( vfit [sll]), 1 )
    
    print(n)
    print(2*Ej/T)
    
   
    
    
    plt.plot(i, abs(V_MQT (i, 0.05 , Ej , Ec , Q, Rn )), '.', label = 'MQT' )
    plt.plot(i, abs(V_PD (i, T , Ej , Ec , Q, Rn )), '.', label = 'PD' )

    plt.plot(i , np.exp(n*i+b) )


#    n, b = polyfit ( i[25:-25] , np.log( V_PD (i, T , Ej , Ec , Q, Rn )[25:-25]), 1 )
#    print(1e3/n)
#    print(b)

#    plt.plot(i, 1.7e-8*np.sinh (i*2.5*2/0.5 ), 'x', label = 'sinh' )


    
#    plt.plot( i, V_AH( i*Ic, Ic , T, Ej, Rn ), 'x',  label = 'AH' )
#    plt.plot( i, V_Likh( i*Ic, Ic , T, Ej, Rn ), 'x',  label = 'likh' )
    R = 1e8
#    plt.plot( v/1e6, I_IZ( v/1e6, Ic = 1 , T=0.5, Ej = 3, Rn = R) , 'x',  label = 'IZ' )
#    plt.plot( v/1e6 , I_IZ( v/1e6, Ic = 1 , T=0.005, Ej = 3, Rn = R ) + v/R*20e8, 'x',  label = 'IZ' )

#    plt.ylim(1e-8,1e-6)

    plt.yscale('log')
    plt.legend()

    plt.subplots()
    plt.plot(i, pNjump(i, Q, Ej, T), '.', label = 'plus' )
    plt.plot(i, mNjump(i, Q, Ej, T), '.', label = 'minus' )
    plt.legend()
    

#    plt.plot(i,  TauTA( i, T, Ej, Ec, Q, pDeltaU(Ej, i), Rn, mode = 'AH'), 'r', i, TauTA( i, T, Ej, Ec, Q, mDeltaU(Ej, i), Rn, mode = 'AH'), '.')

#    plt.plot(i,  np.exp(pDeltaU(Ej, i)/T) - np.exp(mDeltaU(Ej, i)/T), 'rx')
#    plt.plot(i,  (1-i**2)**0.5, 'b.')

#    plt.plot(i,  TauTA( i, T, Ej, Ec, Q, mDeltaU(Ej, i)), 'rx')
#    
#    plt.yscale('log')


#fdir = 'E:\\OneDrive - Rutgers University\\files_py\\expdata\\'
#Frust = 'IV ZF'
#spl = 'chain 68N5\\' + Frust + '\\'    
#    
#plt.plot(i,np.sinh(10*i))

#plt.plot(i,i)

#plt.plot(i,np.exp(i))

#plt.ylim(0.01, 2e3)
#plt.yscale('log')    
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

def dynamics_cp(t,z,f):
    l = 0.5;  # [m]      length of pendulum
    m = 0.5;  # [kg]     mass of pendulum
    M = 0.5;  # [kg]     mass of cart
    b = 0.1;  # [N/m/s]  coefficient of friction between cart and ground
    g = 9.82; # [m/s^2]  acceleration of gravity
    
    if f != 0:
        dz = np.zeros([4,1])
        dz[0] = z[1];
        dz[1] = ( 2*m*l*z[2]**2*np.sin(z[3]) + 3*m*g*np.sin(z[3])*np.cos(z[3]) + 4*f[0](t) - 4*b*z[1] )/( 4*(M+m)-3*m*np.cos(z[3])**2 );
        dz[2] = (-3*m*l*z[2]**2*np.sin(z[3])*np.cos(z[3]) - 6*(M+m)*g*np.sin(z[3]) - 6*(f[0](t)-b*z[1])*np.cos(z[3]) )/( 4*l*(m+M)-3*m*l*np.cos(z[3])**2 );
        dz[3] = z[2];
    else:
        dz = (M+m)*z[1]^2/2 + 1/6*m*l^2*z[2]^2 + m*l*(z[1]*z[2]-g)*np.cos(z[3])/2;

    return dz


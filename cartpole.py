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

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def drawCircle(r,n):
    glBegin(GL_POLYGON);
    for i in range(n):
        x = r * np.cos(2.0 * np.pi * (i/float(n)) );
        y = r * np.sin(2.0 * np.pi * (i/float(n)) );
        glVertex2d(x, y);
    glEnd();			

class CartPole:
    def __init__(self):

        # environment
        self.dt = 0.1#[s]
        self.g = 9.81
        self.stepTime = (int)(self.dt*1000)

        # input
        self.F = 0

        # cart
        self.M = 0.7

        # pole
        self.l = 1
        self.m = 0.325
        self.friction = 0.0

        # variable
        self.reset()

    def reset(self):
        self.x = 0
        self.xd = 0
        self.xdd = 0
        self.th = np.pi + 0.01
        self.thd = 0
        self.thdd = 0
        
    def setState(self,x,th):
        self.x = x
        self.th = th
        
    def simulateSingleStep(self):
        A = np.matrix([[self.m + self.M, self.m * self.l * np.cos(self.th)],
                       [np.cos(self.th), self.l]])
        b = np.matrix([[self.F + self.m * self.l * self.thd**2 * np.sin(self.th)],
                       [-self.friction * self.thd - self.g * np.sin(self.th)]])
        x = np.linalg.inv(A).dot(b)

        self.xdd = x[0, 0]
        self.xd += self.xdd * self.dt
        self.x += self.xd * self.dt

        self.thdd = x[1, 0]
        self.thd += self.thdd * self.dt
        self.th += self.thd * self.dt
        
#        if(self.th < -np.pi):self.th += 2*np.pi
#        if(self.th >  np.pi):self.th -= 2*np.pi
        
    def addForce(self,F):
        self.F = F
        
    def draw(self):
        cartH = 0.1
        cartW = 0.2
        poleW = 0.01
        TireR = 0.05
        
        #floor
        glPushMatrix()
        glTranslated(0,-cartH/2-2*TireR,0)
        glColor3d(0.0, 0.0, 0.0);
        glBegin(GL_QUADS);
        glVertex2d( -5, 0);
        glVertex2d( -5, -0.1);
        glVertex2d(  5, -0.1);
        glVertex2d(  5, 0);
        glEnd();
        glPopMatrix()
    
        #cart
        glTranslated(self.x,0,0)
        glColor3d(1.0, 0.5, 0.0);
        glBegin(GL_QUADS);
        glVertex2d(-cartW, -cartH);
        glVertex2d(-cartW,  cartH);
        glVertex2d( cartW,  cartH);
        glVertex2d( cartW, -cartH);
        glEnd();        
        
        #tire L
        glPushMatrix()
        glTranslated(-cartW+TireR,-cartH/2-TireR,0)
        glColor3d(0.0, 0.0, 0.0);
        drawCircle(TireR,10)
        glPopMatrix()

        #tire R
        glPushMatrix()
        glTranslated(cartW-TireR,-cartH/2-TireR,0)
        glColor3d(0.0, 0.0, 0.0);
        drawCircle(TireR,10)
        glPopMatrix()
        
        #pole
        glRotated(self.th*180/np.pi,0,0,1)
        glColor3d(0.0, 0.5, 0.0);
        glBegin(GL_QUADS);
        glVertex2d( -poleW, 0);
        glVertex2d( -poleW, -self.l);
        glVertex2d(  poleW, -self.l);
        glVertex2d(  poleW, 0);
        glEnd();
        glTranslated(0,-self.l,0)        
        drawCircle(0.1,10)
        
        

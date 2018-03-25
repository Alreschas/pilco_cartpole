# -*- coding: utf-8 -*-

import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

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
        
        

class Drawer:
    def __init__(self):
        self.robo = CartPole() 
        self.latent = np.empty(0)
        self.idleFunc = 0
        self.tMax = 40
        self.t = 0
        self.callTime = 0
        
    def setIdleFunc(self,func):
        self.idleFunc = func
        
    def setLatent(self,latent):
        self.latent = latent
    
    def animation(self,value):
        t = self.t
        
        if(np.size(self.latent) >= self.tMax):
            q1 = self.latent[t,0]
            q2 = self.latent[t,3]
        else:
            q1 = 0
            q2 = 0
#        # set state                    
        self.robo.setState(q1,q2)
#
        glutPostRedisplay()
#        
#        print(self.t)
        self.t += 1
        if(self.t < self.tMax):
            glutTimerFunc(self.robo.stepTime, self.animation, 0);
        else:
            self.t = 0
            if(self.idleFunc != 0):
                self.idleFunc(self.callTime)
            self.callTime += 1
#            glutLeaveMainLoop();
            glutTimerFunc(self.robo.stepTime, self.animation, 0);


    def display(self):
    
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
            
        self.robo.draw()
        
        glFlush()  # enforce OpenGL command

    def resize(self,w, h):
        glViewport(0, 0, w, h);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-w / 200.0, w / 200.0, -h / 200.0, h / 200.0, -1.0, 1.0);
        
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

    def main(self):
        width = 700
        height = 300
        
        glutInitWindowSize(width, height)     # window size
        glutInitWindowPosition(100, 100) # window position
        
        glutInit(sys.argv)
    
#        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);

        glutInitDisplayMode(GLUT_RGBA)

        glutCreateWindow(b"cartPole")      # show window
    
        glutDisplayFunc(self.display)         # draw callback function
        
        glutReshapeFunc(self.resize);
    
        """ initialize """
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        glutTimerFunc(self.robo.stepTime, self.animation, 0);
#            
        glutMainLoop()

if __name__ == "__main__":
    m = Drawer()
    m.main()
    print("end")
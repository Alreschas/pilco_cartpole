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
        self.stepTime = (int)(self.dt*1000)

        # pole
        self.l = 0.5

        # variable
        self.x = 0
        self.th = np.pi + 0.01
        
    def setState(self,x,th):
        self.x = x
        self.th = th
        
    def draw(self):
        glPushMatrix()
        cartH = 0.05
        cartW = 0.1
        poleW = 0.005
        poleR = 0.05
        TireR = 0.025
        
        #floor
        glPushMatrix()
        glTranslated(0,-cartH/2-2*TireR,0)
        glColor3d(0.0, 0.0, 0.0);
        glBegin(GL_QUADS);
        glVertex2d( -2, 0);
        glVertex2d( -2, -0.05);
        glVertex2d(  2, -0.05);
        glVertex2d(  2, 0);
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
        drawCircle(poleR,10)
        
        glPopMatrix()
        
        

class Drawer:
    def __init__(self):
        self.robo = CartPole() 
        self.latent = np.empty(0)
        self.idleFunc = 0
        self.tMax = 40
        self.t = 0
        self.callTime = 1
        
    def setIdleFunc(self,func):
        self.idleFunc = func
    
    def reset(self):
        self.callTime = 0
        
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
        
        self.glut_print(0.6,0.4,GLUT_BITMAP_HELVETICA_18,'epoch:'+str(self.callTime),0,0,0,1)
        self.glut_print(0.6,0.3,GLUT_BITMAP_HELVETICA_18,'step:'+str(self.t),0,0,0,1)
        
        glFlush()  # enforce OpenGL command

    def glut_print(self, x,  y,  font,  text, r,  g , b , a):
    
        blending = False 
        if glIsEnabled(GL_BLEND) :
            blending = True
    
        #glEnable(GL_BLEND)
        glColor3f(r,g,b,a)
        glRasterPos2f(x,y)
        for ch in text :
            glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )
    
        if not blending :
            glDisable(GL_BLEND) 


    def resize(self,w, h):
        glViewport(0, 0, w, h);
        glMatrixMode(GL_PROJECTION)
#        glLoadIdentity();
#        glOrtho(-w / 400.0, w / 400.0, -h / 400.0, h / 400.0, -1.0, 1.0);
        
        s = 400
        glLoadIdentity()
        gluOrtho2D(-w/s, w/s, -h/s, h/s)
        glMatrixMode(GL_MODELVIEW)
        
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
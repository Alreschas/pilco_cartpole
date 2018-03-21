# -*- coding: utf-8 -*-

import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


import cartpole


class Drawer:
    def __init__(self):
        self.robo = cartpole.CartPole()  
        self.t = 0
    
    def animation(self,value):
        t = self.t
        
        q1 = self.latent[0][t,0]
        q2 = self.latent[0][t,3]
        print(q1,q2)

#        q1 = 0
#        q2 = 0
#        # set state                    
        self.robo.setState(q1,q2)
#
        glutPostRedisplay()
#        
#        print(self.t)
        self.t += 1
        if(self.t < 40):
            glutTimerFunc(self.robo.stepTime, self.animation, 0);
        else:
            glutLeaveMainLoop();


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
        


#main
    def main(self,latent):
        width = 700
        height = 300
        
        self.latent = latent
        
                
        glutInitWindowSize(width, height)     # window size
        glutInitWindowPosition(100, 100) # window position
        
        glutInit(sys.argv)
    
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);

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
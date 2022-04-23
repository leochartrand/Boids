import pygame as pg
import compute as co
import util
import datetime
from OpenGL.GL import *
import ctypes

pg.init()

# PyGame parameters
##############################################################################
WIDTH  = (100) * 32
HEIGHT = (100) * 16
SPEED = 3
PARALLEL = 1
gameSettings = util.Settings(WIDTH, HEIGHT, SPEED, PARALLEL)
##############################################################################
SCREEN_SIZE = (gameSettings.WIDTH, gameSettings.HEIGHT)
BACKGROUND_COLOR = (30,30,30)
gameGlock = pg.time.Clock()
pg.display.set_caption("CUDA Boids")
screen = pg.display.set_mode(SCREEN_SIZE, pg.OPENGL|pg.DOUBLEBUF)
##############################################################################
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable( GL_BLEND )
# shader = util.initShader()
# glUseProgram(shader)

co.init(16384, gameSettings)

w = WIDTH/2
h = HEIGHT/2
numIndices = co.numBoids*2
# Main loop
# Based on official docs:
# https://www.pygame.org/docs/ref/draw.html
done = False
while not done:
    for event in pg.event.get(): 
        if event.type == pg.QUIT or event.type == pg.K_ESCAPE: 
              done = True 

    glBegin(GL_POLYGON)
    glColor4f(0.125,0.125,0.125,0.1)
    glVertex2f(-1.0,-1.0)
    glVertex2f(-1.0,1.0)
    glVertex2f(1.0,1.0)
    glVertex2f(1.0,-1.0)
    glEnd()
    
    co.update()

    glBegin(GL_POINTS)

    start_time = datetime.datetime.now()
    array = co.outBuffer
    # vao = glGenVertexArrays(1)
    # glBindVertexArray(vao)
    # vbo = glGenBuffers(1)
    # glBindBuffer(GL_ARRAY_BUFFER, vbo)
    # glBufferData(GL_ARRAY_BUFFER, array.nbytes, array, GL_STATIC_DRAW)
    # glEnableVertexAttribArray(0)
    # glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, 8, ctypes.c_void_p(0))
    # glUseProgram(shader)
    # glBindVertexArray(vao)
    # glColor3f(0.6,1.0,1.0)
    # glDrawArrays(GL_POINTS, 0, numIndices)
    for x,y in zip(array[:,0], array[:,1]):
        # x, y = agent.allo()
        glColor3f(0.6, 1.0, 1.0)
        glVertex2f(x,y)
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    print("draw:",time_diff.total_seconds() * 1000)
    glEnd()

    tickTime = gameGlock.tick()
     
    pg.display.flip()
    glFlush()
# Quits when game is over
# glDeleteProgram(shader)
pg.quit()

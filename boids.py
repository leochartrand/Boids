import pygame as pg
import parallel as par
import sequential as seq
import util
from OpenGL.GL import *
import numpy as np
import ctypes
import datetime

pg.init()

##############################################################################
# Simulation parameters
PARALLEL = True
POPULATION = 32768
WIDTH  = (100) * 36
HEIGHT = (100) * 20
SPEED = 2
COHESION = 0.01
ALIGNMENT = 0.2
SEPARATION = 0.1
NEIGHBOR_DIST = 100
SEPARATION_DIST = 50
gameSettings = util.Settings(POPULATION, SPEED, COHESION, 
    ALIGNMENT, SEPARATION, NEIGHBOR_DIST, SEPARATION_DIST, WIDTH, HEIGHT)
##############################################################################
# PyGame/OpenGL parameters
SCREEN_SIZE = (gameSettings.WIDTH, gameSettings.HEIGHT)
BACKGROUND_COLOR = (30,30,30)
gameGlock = pg.time.Clock()
pg.display.set_caption("CUDA Boids")
screen = pg.display.set_mode(SCREEN_SIZE, pg.OPENGL|pg.DOUBLEBUF)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable( GL_BLEND )
shader = util.initShader()
bgShader = util.initBGShader()
##############################################################################

if PARALLEL:
    par.init(gameSettings)
    renderBuffer = par.renderBuffer
else:
    seq.init(gameSettings)
    renderBuffer = seq.renderBuffer

# Main loop, Based on official docs:
# https://www.pygame.org/docs/ref/draw.html
done = False
while not done:
    for event in pg.event.get(): 
        if event.type == pg.QUIT or event.type == pg.K_ESCAPE: 
              done = True 

    # Update boids
    if PARALLEL:
        par.update()
    else:
        seq.update()

    # Render
    # Background
    glUseProgram(bgShader)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, util.background.nbytes, util.background, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
    glBindVertexArray(vao)
    glDrawArrays(GL_POLYGON, 0, POPULATION) # TODO polygon not accepted
    # Boids
    glUseProgram(shader)
    array = np.copy(renderBuffer)
    array.flatten()
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, array.nbytes, array, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
    glBindVertexArray(vao)
    glDrawArrays(GL_POINTS, 0, POPULATION)

    tickTime = gameGlock.tick()
    print(tickTime)
    pg.display.flip()
    glFlush()
# Quits when game is over
pg.quit()

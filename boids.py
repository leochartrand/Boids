from pickle import POP
import pygame as pg
import numpy as np
import parallel as par
import sequential as seq
import util
from OpenGL.GL import *

pg.init()

##############################################################################
# Simulation parameters
PARALLEL = True
POPULATION = 100000
WIDTH  = (100) * 36
HEIGHT = (100) * 20
SPEED = 2
COHESION = 0.01
ALIGNMENT = 0.2
SEPARATION = 0.1
NEIGHBOR_DIST = 100
SEPARATION_DIST = 50
WRAP_AROUND = True
gameSettings = util.Settings(POPULATION, SPEED, COHESION, ALIGNMENT, 
    SEPARATION, NEIGHBOR_DIST, SEPARATION_DIST, WIDTH, HEIGHT, WRAP_AROUND)
##############################################################################
# PyGame/OpenGL parameters
SCREEN_SIZE = (gameSettings.WIDTH, gameSettings.HEIGHT)
gameGlock = pg.time.Clock()
pg.display.set_caption("CUDA Boids")
screen = pg.display.set_mode(SCREEN_SIZE, pg.OPENGL|pg.DOUBLEBUF)
util.init(POPULATION)
render = util.render
##############################################################################

if PARALLEL:
    par.init(gameSettings)
    renderBuffer = par.renderBuffer
    update = par.update
else:
    seq.init(gameSettings)
    renderBuffer = seq.renderBuffer
    update = seq.update

# Main loop, Based on official docs:
# https://www.pygame.org/docs/ref/draw.html
done = False
while not done:
    for event in pg.event.get(): 
        if event.type == pg.QUIT or event.type == pg.K_ESCAPE: 
              done = True 

    update()

    util.render(renderBuffer)

    tickTime = gameGlock.tick()
    # print(tickTime)

    pg.display.flip()
# Quits when game is over
pg.quit()

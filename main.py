import pygame as pg
import compute as co
import util
import datetime
from OpenGL.GL import *
import OpenGL.GLU

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
screen = pg.display.set_mode(SCREEN_SIZE)
# screen = pg.display.set_mode(SCREEN_SIZE, pg.OPENGL|pg.DOUBLEBUF)
##############################################################################

# Main sprite group
Agents = util.getScenario(util.scenario.CLASSIC)

co.init(Agents, gameSettings)

screen.fill(BACKGROUND_COLOR)
# Fading trails
bg = pg.Surface((WIDTH,HEIGHT))
bg.set_alpha(32)
bg.fill(BACKGROUND_COLOR)
glClearColor(0.125,0.125,0.125,1)

clockFont = pg.font.SysFont("Arial", 50)

# Main loop
# Based on official docs:
# https://www.pygame.org/docs/ref/draw.html
done = False
while not done:
    for event in pg.event.get(): 
        if event.type == pg.QUIT or event.type == pg.K_ESCAPE: 
              done = True 


    # glClear(GL_COLOR_BUFFER_BIT)

    # For pretty fading trails
    screen.blit(bg, (0,0))
    # screen.fill(BACKGROUND_COLOR)
    
    co.update()

    start_time = datetime.datetime.now()
    Agents.update()
    Agents.draw(screen)
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    # print("draw:",time_diff.total_seconds() * 1000)

    tickTime = gameGlock.tick()
    screen.blit(clockFont.render(str(tickTime), 1, pg.Color("white")), (10,0))
              
    pg.display.flip()
# Quits when game is over
pg.quit()

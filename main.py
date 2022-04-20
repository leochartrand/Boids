import pygame as pg
import scenario as sc
import compute as co
import numba as nb
import grid
import util

pg.init()

# PyGame parameters
##############################################################################
WIDTH = 1900
HEIGHT = 1000
SPEED = 3
PARALLEL = 0
gameSettings = util.Settings(WIDTH, HEIGHT, SPEED, PARALLEL)
##############################################################################
SCREEN_SIZE = (gameSettings.WIDTH, gameSettings.HEIGHT)
BACKGROUND_COLOR = (50,50,50)
BACKGROUND_COLOR2 = (50,50,50,50)
gameGlock = pg.time.Clock()
pg.display.set_caption("CUDA Boids")
screen = pg.display.set_mode(SCREEN_SIZE)
##############################################################################

# Groupe maitre
Agents = sc.getScenario(sc.type.THOUSAND, gameSettings)

co.init(Agents, gameSettings)

g = grid.init(WIDTH,HEIGHT,Agents.sprites())

# Main loop
# Based on official docs:
# https://www.pygame.org/docs/ref/draw.html
done = False
while not done:
    for event in pg.event.get(): 
        if event.type == pg.QUIT or event.type == pg.K_ESCAPE: 
              done = True 

    tickTime = gameGlock.tick()
    print(tickTime)
    screen.fill(BACKGROUND_COLOR)
    
    # co.update()
    Agents.update()
    Agents.draw(screen)
              
    pg.display.flip()
# Quits when game is over
pg.quit()

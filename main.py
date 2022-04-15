import pygame
import cupy
import util
import scenario as sc

pygame.init()

# PyGame parameters
##############################################################################
WIDTH = 1920
HEIGHT = 1040
SPEED = 3
##############################################################################
gameSettings = util.Settings(WIDTH, HEIGHT, SPEED)
SCREEN_SIZE = (gameSettings.WIDTH, gameSettings.HEIGHT)
BACKGROUND_COLOR = (50,50,50)
gameGlock = pygame.time.Clock()
pygame.display.set_caption("CUDA Boids")
screen = pygame.display.set_mode(SCREEN_SIZE)

# Groupe maitre
Agents = sc.getScenario(sc.type.CLASSIC, gameSettings)

# Main loop
# Based on official docs:
# https://www.pygame.org/docs/ref/draw.html
done = False
while not done:
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT or event.type == pygame.K_ESCAPE: 
              done = True 

    gameGlock.tick(100)
    screen.fill(BACKGROUND_COLOR)
    
    Agents.update()
    Agents.draw(screen)
              
    pygame.display.flip()
# Quits when game is over
pygame.quit()



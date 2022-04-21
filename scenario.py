import util
from enum import Enum
import pygame
import sprites

class type(Enum):
    CLASSIC = 0
    TWO = 1
    HUNT = 2
    FEAST = 3
    BIG_CLASSIC = 4
    THOUSAND = 5

def getScenario(index):
    Agents = pygame.sprite.Group()
    if index == type.CLASSIC:
        Greens = pygame.sprite.Group()
        for index in range(100):
            Greens.add(sprites.Sprite(util.Species.CYAN, index))
        Agents.add(Greens.sprites())
        return Agents

    if index == type.TWO:
        Greens = pygame.sprite.Group()
        for index in range(500):
            Greens.add(sprites.Sprite(util.Species.GREEN, index))
        Agents.add(Greens.sprites())
        Oranges = pygame.sprite.Group()
        for index in range(500, 1000):
            Oranges.add(sprites.Sprite(util.Species.ORANGE, index))
        Agents.add(Oranges.sprites())
        return Agents


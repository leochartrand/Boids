from dataclasses import dataclass
from enum import Enum
import pygame as pg
import sprites

@dataclass
class Settings:
    WIDTH : int
    HEIGHT: int
    SPEED: int
    PARALLEL: int

class Background(pg.sprite.Sprite):
    def __init__(self, color, width, height):
        super().__init__()
        self.image = pg.Surface([width, height])
        self.image.fill(color)

class scenario(Enum):
    CLASSIC = 0
    TWO = 1

def getScenario(index):
    Agents = pg.sprite.Group()
    if index == scenario.CLASSIC:
        Greens = pg.sprite.Group()
        for index in range(4096): # ideally a multiple of1024
            Greens.add(sprites.Sprite(sprites.Species.CYAN, index))
        Agents.add(Greens.sprites())
        return Agents

    if index == scenario.TWO:
        Greens = pg.sprite.Group()
        for index in range(2048):
            Greens.add(sprites.Sprite(sprites.Species.GREEN, index))
        Agents.add(Greens.sprites())
        Oranges = pg.sprite.Group()
        for index in range(2048, 4096):
            Oranges.add(sprites.Sprite(sprites.Species.ORANGE, index))
        Agents.add(Oranges.sprites())
        return Agents



from dataclasses import dataclass
import pygame as pg
from enum import Enum

@dataclass
class Settings:
    WIDTH : int
    HEIGHT: int
    SPEED: int
    PARALLEL: int

class Species(Enum):
    GREEN = 0
    ORANGE = 1
    CYAN = 2

def getSpeciesImage(index):
    if index == Species.GREEN:
        return pg.image.load("assets/boid_green.png")
    if index == Species.ORANGE:
        return pg.image.load("assets/boid_orange.png")
    if index == Species.CYAN:
        return pg.image.load("assets/boid_cyan.png")

class Background(pg.sprite.Sprite):
    def __init__(self, color, width, height):
        super().__init__()
        self.image = pg.Surface([width, height])
        self.image.fill(color)



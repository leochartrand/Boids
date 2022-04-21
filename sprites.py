import pygame as pg
import compute as co
from enum import Enum

class Sprite (pg.sprite.Sprite):
    def __init__(self, species, index):
        super().__init__()
        self.index = index
        self.image = getSpeciesImage(species)
        self.rect = self.image.get_rect()
        self.imgBase = pg.transform.scale(pg.transform.rotate(self.image, -90), (10, 10)) #used for updating (rotation)
        self.species = species
    
    def update(self):
        x, y, dx, dy = co.getPosAndDir(self.index)
        self.rect.center = pg.Vector2(x,y)
        angle = pg.math.Vector2(1,0).angle_to(pg.Vector2(dx,dy))
        self.image = pg.transform.rotate(self.imgBase, -angle)

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
    


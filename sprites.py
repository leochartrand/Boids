import pygame as pg
import random as rd
import util
import compute as co

class Sprite (pg.sprite.Sprite):
    def __init__(self, species, index):
        super().__init__()
        self.index = index
        self.image = util.getSpeciesImage(species)
        self.rect = self.image.get_rect()
        self.imgBase = pg.transform.scale(pg.transform.rotate(self.image, -90), (10, 10)) #used for updating (rotation)
        self.species = species
    
    def update(self):
        x, y, dx, dy = co.getPosAndDir(self.index)
        self.rect.center = pg.Vector2(x,y)
        angle = pg.math.Vector2(1,0).angle_to(pg.Vector2(dx,dy))
        self.image = pg.transform.rotate(self.imgBase, -angle)

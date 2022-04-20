from sklearn import neighbors
import util
import pygame as pg
import cupy as cp
import numpy as np
import numba as nb
import random as rd
import math

def init(Agents, settings):
    global WIDTH, HEIGHT, SPEED, PARALLEL
    WIDTH = settings.WIDTH
    HEIGHT = settings.HEIGHT
    SPEED = settings.SPEED
    PARALLEL = settings.PARALLEL
    #
    global x_gpu, posDir, boids, numBoids, gpu_species, dataBuffer
    boids = Agents.sprites()
    numBoids = len(boids)
    posDir = np.zeros((numBoids, 19), dtype=float)
    dataBuffer = np.zeros((numBoids, 2), dtype=float)
    for i in range(numBoids):
        posDir[i,0] = rd.uniform(0, WIDTH)
        posDir[i,1] = rd.uniform(0, HEIGHT)
        tempDir = normalize(rd.random()*2-1, rd.random()*2-1)
        posDir[i,2] = rd.uniform(0, tempDir[0])
        posDir[i,3] = rd.uniform(0, tempDir[1])
        posDir[i,18] = boids[i].species.value
    # print(np.shape(x_cpu))
    # print(x_cpu)
    species = np.array([0.0,1.0,2.0,3.0])
    gpu_species = cp.asarray(species)
    # gpu_species = gpu_species ** 2
    # cpu_species = cp.asnumpy(gpu_species)
    # print(cpu_species)

def getPosAndDir(index):
    position = pg.Vector2(posDir[index,0], posDir[index,1])
    direction = pg.Vector2(posDir[index,2], posDir[index,3])
    return position, direction

def sendData():
    pass

def retrieveData():
    pass

def update():
    global posDir, x_gpu
    if PARALLEL:
        pass
    else:
        for i in nb.prange(numBoids):
            # x_gpu = cp.asarray(x_cpu)
            x_gpu = posDir
            getNewValues(i)
            adjustDirection(i)
            clampSpeed(i)
            move(i)
            wrapAround(i)
            # x_cpu = cp.asnumpy(x_gpu)
            posDir = x_gpu

##############################################################################
# SEQUENTIAL
##############################################################################

def move(index):
    global x_gpu
    x_gpu[index,0] += x_gpu[index,2] * SPEED
    x_gpu[index,1] += x_gpu[index,3] * SPEED

def normalize(x, y):
    l = length(x,y)
    return x / l, y / l

def length(x,y):
    return cp.sqrt(cp.square(x) + cp.square(y))

def distance(a,b):
    global x_gpu
    return abs(length(x_gpu[a,0] - x_gpu[b,0], x_gpu[a,1] - x_gpu[b,1]))

def wrapAround(index):
    global x_gpu
    if x_gpu[index,0] > WIDTH:
        x_gpu[index,0] = 0
    elif x_gpu[index,0] < 0 :
        x_gpu[index,0] = WIDTH
    if x_gpu[index,1] > HEIGHT:
        x_gpu[index,1] = 0
    elif x_gpu[index,1] < 0 :
        x_gpu[index,1] = HEIGHT

def clampSpeed(index):
    global x_gpu
    l = length(x_gpu[index,2], x_gpu[index,3])
    if l > 1.0:
        x_gpu[index,2], x_gpu[index,3] = normalize(x_gpu[index,2], x_gpu[index,3])
    elif l < 0.5:
        x, y = normalize(x_gpu[index,2], x_gpu[index,3])
        x_gpu[index,2] = x / 2
        x_gpu[index,3] = y / 2

def adjustDirection(index):
    global x_gpu
    x, y = 0.0, 0.0
    # If it's a normal boid
    if x_gpu[index,18] != gpu_species[3]:
        # Alignment
        x += x_gpu[index,4] / 10
        y += x_gpu[index,5] / 10
        # Cohesion
        x += x_gpu[index,6] / 1000
        y += x_gpu[index,7] / 1000
        # Separation
        x += x_gpu[index,8]
        y += x_gpu[index,9]
        # Cohesion - Strangers
        x -= x_gpu[index,10] / 100
        y -= x_gpu[index,11] / 100
        # Separation - Strangers
        x += x_gpu[index,12] * 10
        y += x_gpu[index,13] * 10
        # Cohesion - Predators
        x -= x_gpu[index,14]
        y -= x_gpu[index,15]
        # Separation - Predators
        x += x_gpu[index,16] * 10
        y += x_gpu[index,17] * 10
    # If it's a predator
    else:
        x += x_gpu[index,10] / 10
        y += x_gpu[index,11] / 10
    x_gpu[index,2] += x
    x_gpu[index,3] += y
    

def getNewValues(index):
    global x_gpu
    friends, strangers, predators = getNeighbors(index)
    if len(friends) > 0:
        # a,b,c,d,e,f = calculate(index, friends)
        x_gpu[index,4], x_gpu[index,5] = alignment(friends)
        x_gpu[index,6], x_gpu[index,7] = cohesion(index,friends)
        x_gpu[index,8], x_gpu[index,9] = separation(index,friends)
    if len(strangers) > 0:
        posDir[index,10], posDir[index,11] = cohesion(index,strangers)
        posDir[index,12], posDir[index,13] = separation(index,strangers)
    if len(predators) > 0:
        posDir[index,14], posDir[index,15] = cohesion(index,predators)
        posDir[index,16], posDir[index,17] = separation(index,predators)

def getNeighbors(index):
    global x_gpu
    friends, strangers, predators = [], [], []
    for neighbor in range(numBoids):
        if neighbor != index and distance(index, neighbor) < 100.0:
            if x_gpu[neighbor,18] == x_gpu[index,18]:
                friends.append(neighbor)
            elif x_gpu[neighbor,18] == gpu_species[3]:
                predators.append(neighbor)
            else:
                strangers.append(neighbor)
    return friends, strangers, predators

def cohesion(index, neighbors):
    global x_gpu
    cohesionX, cohesionY = 0.0, 0.0
    for agent in neighbors:
        cohesionX += x_gpu[agent, 0]
        cohesionY += x_gpu[agent, 1]
    cohesionX -= x_gpu[index, 0]
    cohesionY -= x_gpu[index, 1]
    cohesionX, cohesionY = normalize(cohesionX, cohesionY)
    return cohesionX, cohesionY

def alignment(neighbors):
    global x_gpu
    alignmentX, alignmentY = 0.0, 0.0
    for agent in neighbors:
        alignmentX += x_gpu[agent, 2]
        alignmentY += x_gpu[agent, 3]
    alignmentX, alignmentY = normalize(alignmentX, alignmentY)
    return alignmentX, alignmentY

def separation(index, neighbors):
    global x_gpu
    separationX, separationY = 0.0, 0.0
    numSeparation = 0
    for agent in neighbors:
        distX = x_gpu[index,0] - x_gpu[agent,0]
        distY = x_gpu[index,1] - x_gpu[agent,1]
        distLength = length(distX, distY)
        sqrt = length(distX, distY) ** 2
        if distLength < 40:
            if distLength > 0:
                distX /= sqrt
                distY /= sqrt
            separationX += distX
            separationY += distY
            numSeparation += 1
    return separationX, separationY

def calculate(index, neighbors):
    global x_gpu
    cohesionX, cohesionY = 0.0, 0.0
    alignmentX, alignmentY = 0.0, 0.0
    separationX, separationY = 0.0, 0.0
    for agent in neighbors:
        cohesionX += x_gpu[agent, 0]
        cohesionY += x_gpu[agent, 1]
        alignmentX += x_gpu[agent, 2]
        alignmentY += x_gpu[agent, 3]
        distX = x_gpu[index,0] - x_gpu[agent,0]
        distY = x_gpu[index,1] - x_gpu[agent,1]
        distLength = length(distX, distY)
        sqrt = length(distX, distY) ** 2
        if distLength < 40:
            separationX += distX / sqrt
            separationY += distY / sqrt
    cohesionX -= x_gpu[index, 0]
    cohesionY -= x_gpu[index, 1]
    cohesionX, cohesionY = normalize(cohesionX, cohesionY)
    alignmentX, alignmentY = normalize(alignmentX, alignmentY)
    return alignmentX, alignmentY, cohesionX, cohesionY, separationX, separationY






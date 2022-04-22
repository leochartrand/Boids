import numpy as np
import numba as nb
import random as rd

def init(Agents, settings):
    global WIDTH, HEIGHT, SPEED, PARALLEL, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT
    WIDTH = settings.WIDTH
    HEIGHT = settings.HEIGHT
    SPEED = settings.SPEED
    PARALLEL = settings.PARALLEL
    GRID_CELL_SIZE = 100
    GRID_WIDTH = WIDTH//GRID_CELL_SIZE
    GRID_HEIGHT = HEIGHT//GRID_CELL_SIZE
    #
    global boids, numBoids, dataBuffer, tempBuffer
    boids = Agents.sprites()
    numBoids = len(boids)
    dataBuffer = np.zeros((numBoids, 4), dtype=float)
    tempBuffer = np.zeros((numBoids, 4), dtype=float)
    for i in range(numBoids):
        dataBuffer[i,0] = rd.uniform(0, WIDTH)
        dataBuffer[i,1] = rd.uniform(0, HEIGHT)
        dataBuffer[i,2] = rd.random()*2-1
        dataBuffer[i,3] = rd.random()*2-1
    # Grid initialization
    global boidGrid, neighborGrid
    boidGrid = np.empty((GRID_WIDTH*GRID_HEIGHT,), dtype=object)
    for i,j in enumerate(boidGrid): 
        boidGrid[i] = []
    neighborGrid = np.empty((GRID_WIDTH*GRID_HEIGHT,), dtype=object)
    for i,j in enumerate(neighborGrid): 
        neighborGrid[i] = []

def getPosAndDir(index):
    return dataBuffer[index,0],dataBuffer[index,1],dataBuffer[index,2],dataBuffer[index,3]

#######################################
# Update
#######################################
# @nb.jit
def update():
    global dataBuffer, tempBuffer
    tempBuffer[:,0:4] = dataBuffer[:,0:4]
    updateGrids()
    for i in nb.prange(numBoids):
        updateBoid(i)
    dataBuffer[:,0:4] = tempBuffer[:,0:4]

def updateGrids():
    global boidGrid, neighborGrid
    for index, cell in enumerate(boidGrid):
        cell.clear()
    for boidIndex in range(numBoids): 
        x = int(tempBuffer[boidIndex,0]//GRID_CELL_SIZE) % GRID_WIDTH
        y = int(tempBuffer[boidIndex,1]//GRID_CELL_SIZE) % GRID_HEIGHT
        gridIndex = x + y * GRID_WIDTH
        boidGrid[gridIndex].append(boidIndex)
    for index, cell in enumerate(neighborGrid):
        x = index % GRID_WIDTH
        y = index //GRID_HEIGHT
        cell.clear()
        tileOffset = {-1, 0, 1}
        for i in tileOffset:
            for j in tileOffset:
                gridIndex = (int(x//GRID_CELL_SIZE) + i)%GRID_WIDTH + (int(y//GRID_CELL_SIZE) + j)%GRID_HEIGHT * GRID_WIDTH
                cell.append(boidGrid[gridIndex])


def updateBoid(index):
    global tempBuffer, boidGrid, neighborGrid
    # READ SHARED MEMORY - NO WRITING
    # USE LOCAL MEMORY
    x, y = tempBuffer[index,0], tempBuffer[index,1]
    dx, dy = tempBuffer[index,2], tempBuffer[index,3]
    # Get cell neighbors
    neighbors = []
    gridIndex = int(x//GRID_CELL_SIZE)%GRID_WIDTH + int(y//GRID_CELL_SIZE)%GRID_HEIGHT * GRID_WIDTH
    for agent in boidGrid[gridIndex]:
        if agent != index and np.sqrt((tempBuffer[index,0] - tempBuffer[agent,0])**2 + (tempBuffer[index,1] - tempBuffer[agent,1])**2) < 100:
            neighbors.append(agent)
    # Apply boid rules
    if len(neighbors) > 0:
        # Alignment
        alignmentX, alignmentY = 0.0, 0.0
        for agent in neighbors:
            alignmentX += tempBuffer[agent, 2]
            alignmentY += tempBuffer[agent, 3]
        # Normalize alignment vector
        l = np.sqrt(alignmentX**2 + alignmentY**2)
        alignmentX, alignmentY = alignmentX / l, alignmentY / l
        dx += alignmentX / 10
        dy += alignmentY / 10
        # Cohesion
        cohesionX, cohesionY = 0.0, 0.0
        for agent in neighbors:
            cohesionX += tempBuffer[agent, 0]
            cohesionY += tempBuffer[agent, 1]
        cohesionX -= tempBuffer[index, 0]
        cohesionY -= tempBuffer[index, 1]
        # Normalize cohesion vector
        l = np.sqrt(cohesionX**2 + cohesionY**2)
        cohesionX, cohesionY = cohesionX / l, cohesionY / l
        dx += cohesionX / 1000
        dy += cohesionY / 1000
        # Separation
        separationX, separationY = 0.0, 0.0
        numSeparation = 0
        for agent in neighbors:
            distX = tempBuffer[index,0] - tempBuffer[agent,0]
            distY = tempBuffer[index,1] - tempBuffer[agent,1]
            distLength = np.sqrt(distX**2 + distY**2)
            sqrt = distLength ** 2
            if distLength < 40:
                if distLength > 0:
                    distX /= sqrt
                    distY /= sqrt
                separationX += distX
                separationY += distY
                numSeparation += 1
        if numSeparation > 0:
            separationX /= numSeparation
            separationY /= numSeparation
        dx += separationX
        dy += separationY
    # WRITE TO SHARED MEMORY - WAIT SYNC
    # Clamp speed, Update directions
    l = np.sqrt(dx**2 + dy**2)
    tempBuffer[index,2], tempBuffer[index,3] = dx / l, dy / l
    # Move position
    tempBuffer[index,0] += tempBuffer[index,2] * SPEED
    tempBuffer[index,1] += tempBuffer[index,3] * SPEED
    # Edge wrap
    if tempBuffer[index,0] > WIDTH:
        tempBuffer[index,0] = 0
    elif tempBuffer[index,0] < 0 :
        tempBuffer[index,0] = WIDTH
    if tempBuffer[index,1] > HEIGHT:
        tempBuffer[index,1] = 0
    elif tempBuffer[index,1] < 0 :
        tempBuffer[index,1] = HEIGHT




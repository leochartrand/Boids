import numpy as np
import numba as nb
import random as rd
from numba import cuda
import math
import datetime

def init(n, settings):
    global WIDTH, HEIGHT,w,h, SPEED, PARALLEL, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT
    WIDTH = settings.WIDTH
    HEIGHT = settings.HEIGHT
    w = WIDTH/2
    h = HEIGHT/2
    SPEED = settings.SPEED
    PARALLEL = settings.PARALLEL
    GRID_CELL_SIZE = 100
    GRID_WIDTH = WIDTH//GRID_CELL_SIZE
    GRID_HEIGHT = HEIGHT//GRID_CELL_SIZE
    #
    global boids, numBoids, outBuffer, boidBuffer
    numBoids = n
    outBuffer = np.zeros((numBoids, 2), dtype=float)
    boidBuffer = np.zeros((numBoids, 4), dtype=float)
    for i in range(numBoids):
        boidBuffer[i,0] = rd.uniform(0, WIDTH)
        boidBuffer[i,1] = rd.uniform(0, HEIGHT)
        boidBuffer[i,2] = rd.random()*2-1
        boidBuffer[i,3] = rd.random()*2-1
    # Grid initialization
    global boidGrid, neighborGrid
    boidGrid = np.empty((GRID_WIDTH*GRID_HEIGHT,), dtype=object)
    for i,j in enumerate(boidGrid): 
        boidGrid[i] = []
    neighborGrid = np.empty((GRID_WIDTH*GRID_HEIGHT,128), dtype=int)
    neighborGrid.fill(-1)

def getPosAndDir(index):
    return outBuffer[index,0],outBuffer[index,1]

#######################################
# Update
#######################################
def update():
    global outBuffer, boidBuffer
    # inBuffer[:,0:4] = outBuffer[:,0:4]
    updateGrids()
    # start_time = datetime.datetime.now()
    if PARALLEL == 1:
        nthreads = 1024
        nblocks = numBoids // nthreads
        getBoiddata[nblocks, nthreads](boidBuffer, outBuffer, neighborGrid, numBoids)
    else:
        for i in range(numBoids):
            # updateBoid(i)
            pass
    # end_time = datetime.datetime.now()
    # time_diff = (end_time - start_time)
    # print("para:",time_diff.total_seconds() * 1000)

def updateGrids():
    start_time = datetime.datetime.now()
    global boidGrid, neighborGrid
    for index, cell in enumerate(boidGrid):
        cell.clear()
    for boidIndex in range(numBoids): 
        x = int(boidBuffer[boidIndex,0]//GRID_CELL_SIZE) % GRID_WIDTH
        y = int(boidBuffer[boidIndex,1]//GRID_CELL_SIZE) % GRID_HEIGHT
        gridIndex = x + y * GRID_WIDTH
        boidGrid[gridIndex].append(boidIndex)
    neighborGrid.fill(-1)
    for index, cell in enumerate(neighborGrid):
        x = index % GRID_WIDTH
        y = index //GRID_WIDTH
        # Get neighbor cells
        list = []
        tileOffset = {-1, 0, 1}
        for i in tileOffset:
            for j in tileOffset:
                gridIndex = (x + i)%GRID_WIDTH + (y + j)%GRID_HEIGHT * GRID_WIDTH
                list.extend(boidGrid[gridIndex])
        rd.shuffle(list)
        # print(len(list))
        for i in range(len(list)):
            if i >= 128:
                break
            cell[i] = list[i]
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    print("grid:",time_diff.total_seconds() * 1000)

#######################################
# Parallel
#######################################
@cuda.jit
def getBoiddata(boidBuffer, outBuffer, neighborGrid, numBoids):
    index = cuda.grid(1)
    if index >= numBoids:
        return
    # READ SHARED MEMORY - NO WRITING
    # USE LOCAL MEMORY
    x, y = boidBuffer[index,0], boidBuffer[index,1]
    dx, dy = boidBuffer[index,2], boidBuffer[index,3]
    numNeighbors = 0
    alignmentX, alignmentY = 0.0, 0.0
    cohesionX, cohesionY = 0.0, 0.0
    separationX, separationY = 0.0, 0.0
    numSeparation = 0
    # Get cell neighbors
    gridIndex = int(x//GRID_CELL_SIZE)%GRID_WIDTH + int(y//GRID_CELL_SIZE)%GRID_HEIGHT * GRID_WIDTH
    for agent in neighborGrid[gridIndex]:
        # No more neighbors in array
        if agent == -1:
            break
        if agent != index and math.sqrt((boidBuffer[index,0] - boidBuffer[agent,0])**2 + (boidBuffer[index,1] - boidBuffer[agent,1])**2) < 100:
            numNeighbors += 1
            alignmentX += boidBuffer[agent, 2]
            alignmentY += boidBuffer[agent, 3]
            cohesionX += boidBuffer[agent, 0]
            cohesionY += boidBuffer[agent, 1]
            distX = x - boidBuffer[agent,0]
            distY = y - boidBuffer[agent,1]
            distLength = math.sqrt(distX**2 + distY**2)
            sqrt = distLength ** 2
            if distLength < 30:
                if distLength > 0:
                    distX /= sqrt
                    distY /= sqrt
                separationX += distX
                separationY += distY
                numSeparation += 1
    # Apply boid rules
    # Alignment
    alignmentX /= numNeighbors
    alignmentY /= numNeighbors
    # Normalize alignment vector
    l = math.sqrt(alignmentX**2 + alignmentY**2)
    alignmentX, alignmentY = alignmentX / l, alignmentY / l
    # Add alignment vector to direction
    dx += alignmentX / 10
    dy += alignmentY / 10
    # Cohesion
    cohesionX /= numNeighbors
    cohesionY /= numNeighbors
    # Get vector towards center of mass
    cohesionX -= x
    cohesionY -= y
    # Normalize cohesion vector
    l = math.sqrt(cohesionX**2 + cohesionY**2)
    cohesionX, cohesionY = cohesionX / l, cohesionY / l
    # Add cohesion vector to direction
    dx += cohesionX / 50
    dy += cohesionY / 50
    # Separation
    if numSeparation > 0:
        separationX /= numSeparation
        separationY /= numSeparation
    # Add separation vector to direction
    dx += separationX * 5
    dy += separationY * 5
    # Update directions
    dx = boidBuffer[index,2] + dx * 3 # Weight w, ratio of 1:w with old/new direction
    dy = boidBuffer[index,3] + dy * 3
    # Clamp speed
    l = math.sqrt(dx**2 + dy**2)
    dx, dy = dx / l, dy / l
    # Move position
    x += dx * SPEED
    y += dy * SPEED
    # Edge wrap
    if x > WIDTH:
        x = 0
    elif x < 0 :
        x = WIDTH
    if y > HEIGHT:
        y = 0
    elif y < 0 :
        y = HEIGHT
    # Render coordinates
    rx = x - w
    ry = y - h
    rx /= w
    ry /= h
    # WRITE TO SHARED MEMORY - WAIT SYNC
    cuda.syncthreads()
    outBuffer[index,0] = rx
    outBuffer[index,1] = ry
    boidBuffer[index,0] = x
    boidBuffer[index,1] = y
    boidBuffer[index,2] = dx
    boidBuffer[index,3] = dy


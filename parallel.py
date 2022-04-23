import numpy as np
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
    global numBoids, renderBuffer, inBuffer, outBuffer, lookUpTable, tileIndexTable, boidTable, tileOffsetTable
    numBoids = n
    renderBuffer = np.zeros((numBoids, 2), dtype=np.float32)
    inBuffer = np.zeros((numBoids, 4), dtype=np.float32)
    outBuffer = np.zeros((numBoids, 4), dtype=np.float32)
    for i in range(numBoids):
        inBuffer[i,0] = rd.uniform(0, WIDTH)
        inBuffer[i,1] = rd.uniform(0, HEIGHT)
        inBuffer[i,2] = rd.random()*2-1
        inBuffer[i,3] = rd.random()*2-1
    tileIndexTable = np.zeros(GRID_WIDTH*GRID_HEIGHT, dtype=np.int32)
    boidTable = np.zeros((numBoids,2), dtype=np.int32)
    for index, cell in enumerate(boidTable):
        cell[0] = index
    # Look-Up table
    temp = np.zeros((GRID_WIDTH*GRID_HEIGHT,9), dtype=int)
    tileOffset = {-1, 0, 1}
    for index, cell in enumerate(temp):
        x = index % GRID_WIDTH
        y = index //GRID_WIDTH
        col = 0
        for i in tileOffset:
            for j in tileOffset:
                cell[col] = (x + i)%GRID_WIDTH + (y + j)%GRID_HEIGHT * GRID_WIDTH
                col += 1
    lookUpTable = cuda.to_device(temp)
    # Tile Offset Table (for distance check w/ edge wrapping)
    tempTileOffsetTable = np.empty((9,2), dtype=np.int32)
    tileValues = (-1, 0, 1)
    totIndex = 0
    for i in tileValues:
        for j in tileValues:
            tempTileOffsetTable[totIndex,0] = i * GRID_WIDTH
            tempTileOffsetTable[totIndex,1] = j * GRID_HEIGHT
            totIndex += 1
    tileOffsetTable = cuda.to_device(tempTileOffsetTable)
    

#######################################
# Update
#######################################
def update():
    global renderBuffer, inBuffer, neighborGrid, lookUpTable, boidTable, outBuffer, tileOffsetTable, tileIndexTable
    start_time = datetime.datetime.now()
    if PARALLEL == 1:
        nthreads = 1024
        nblocks = numBoids // nthreads
        fillBoidTable[nblocks, nthreads](inBuffer, boidTable, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT)
        boidTable = boidTable[boidTable[:,1].argsort()]
        fillTileIndexTable()
        getBoiddata[nblocks, nthreads](inBuffer, outBuffer, renderBuffer, numBoids, lookUpTable, tileIndexTable, boidTable, tileOffsetTable)
        inBuffer = outBuffer
    else:
        for i in range(numBoids):
            pass
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    print("para:",time_diff.total_seconds() * 1000)

def fillTileIndexTable():
    global tileIndexTable, boidTable
    tileIndexTable.fill(-1)
    currentIndex = -1
    for index, cell in enumerate(boidTable):
        if currentIndex != cell[1]:
            currentIndex = cell[1]
            tileIndexTable[currentIndex] = index

#######################################
# Parallel
#######################################
@cuda.jit
def fillBoidTable(inBuffer, boidTable, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT):
    index = cuda.grid(1)
    x = int(inBuffer[index,0]//GRID_CELL_SIZE) % GRID_WIDTH
    y = int(inBuffer[index,1]//GRID_CELL_SIZE) % GRID_HEIGHT
    gridIndex = x + y * GRID_WIDTH
    boidTable[index,1] = gridIndex

@cuda.jit
def getBoiddata(inBuffer, outBuffer, renderBuffer, numBoids, lookUpTable, tileIndexTable, boidTable, tileOffsetTable):
    index = cuda.grid(1)
    if index >= numBoids:
        return
    # READ SHARED MEMORY - NO WRITING
    # USE LOCAL MEMORY
    x, y = inBuffer[index,0], inBuffer[index,1]
    dx, dy = inBuffer[index,2], inBuffer[index,3]
    numNeighbors = 0
    alignmentX, alignmentY = 0.0, 0.0
    cohesionX, cohesionY = 0.0, 0.0
    separationX, separationY = 0.0, 0.0
    numSeparation = 0
    #Check if close to edge (for distanche check)
    if x < 100 or y < 100 or WIDTH - x < 100 or HEIGHT - y < 100:
        onEdge = True
    else: 
        onEdge = False
    # Get cell neighbors
    gridIndex = int(x//GRID_CELL_SIZE)%GRID_WIDTH + int(y//GRID_CELL_SIZE)%GRID_HEIGHT * GRID_WIDTH
    for tile in lookUpTable[gridIndex]:
        startIndex = tileIndexTable[tile]
        if startIndex != -1:
            value = tile
            while value == tile and startIndex < numBoids:
                agent = boidTable[startIndex,0]
                startIndex += 1
                value = boidTable[startIndex,1]
                ax = inBuffer[agent,0]
                ay = inBuffer[agent,1]
                adx = inBuffer[agent,2]
                ady = inBuffer[agent,3]
                if onEdge:
                    isNb = isNeighborEdge(x,y,ax,ay,tileOffsetTable)
                else: 
                    isNb = (math.sqrt((x - ax)**2 + (y - ay)**2) < 100)
                if agent != index and isNb:
                    numNeighbors += 1
                    alignmentX += adx
                    alignmentY += ady
                    cohesionX += ax
                    cohesionY += ay
                    distX = x - ax
                    distY = y - ay
                    distLength = math.sqrt(distX**2 + distY**2)
                    sqrt = distLength ** 2
                    if distLength < 50:
                        if distLength > 0:
                            distX /= sqrt
                            distY /= sqrt
                        separationX += distX
                        separationY += distY
                        numSeparation += 1
    if numNeighbors > 0:
        # Apply boid rules
        # Alignment
        alignmentX /= numNeighbors
        alignmentY /= numNeighbors
        # Normalize alignment vector
        l = math.sqrt(alignmentX**2 + alignmentY**2)
        alignmentX, alignmentY = alignmentX / l, alignmentY / l
        # Add alignment vector to direction
        dx += alignmentX / 5
        dy += alignmentY / 5
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
        dx += cohesionX / 100
        dy += cohesionY / 100
        # Separation
        if numSeparation > 0:
            separationX /= numSeparation
            separationY /= numSeparation
        # Add separation vector to direction
        dx += separationX * 8
        dy += separationY * 8
    # Update directions
    dx = inBuffer[index,2] + dx * 5 # Weight w, ratio of 1:w with old/new direction
    dy = inBuffer[index,3] + dy * 5
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
    renderBuffer[index,0] = rx
    renderBuffer[index,1] = ry
    outBuffer[index,0] = x
    outBuffer[index,1] = y
    outBuffer[index,2] = dx
    outBuffer[index,3] = dy

@cuda.jit(device=True)
def isNeighborEdge(x,y,ax,ay,tileOffsetTable):
    tileValues = (-1, 0, 1)
    for offset in tileOffsetTable:
            if math.sqrt((x - (ax + offset[0]))**2 + (y - (ay + offset[1]))**2) < 100:
                return True
    return False


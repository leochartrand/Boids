from numba import cuda
import numpy as np
import cupy as cp
import random as rd
import math

def init(settings):
    #######################################
    # Constants
    #######################################
    global POPULATION, COHESION, ALIGNMENT, SEPARATION, NEIGHBOR_DIST_SQUARED, SEPARATION_DIST_SQUARED, WRAP_AROUND
    global WIDTH, HEIGHT, HALF_WIDTH, HALF_HEIGHT, SPEED, PARALLEL, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT
    POPULATION = settings.POPULATION
    SPEED = settings.SPEED
    COHESION = settings.COHESION
    ALIGNMENT = settings.ALIGNMENT
    SEPARATION = settings.SEPARATION
    NEIGHBOR_DIST_SQUARED = settings.NEIGHBOR_DIST ** 2 # Useful for evaluating distances
    SEPARATION_DIST_SQUARED = settings.SEPARATION_DIST ** 2
    WIDTH = settings.WIDTH
    HEIGHT = settings.HEIGHT
    HALF_WIDTH = WIDTH/2
    HALF_HEIGHT = HEIGHT/2
    GRID_CELL_SIZE = settings.NEIGHBOR_DIST
    GRID_WIDTH = WIDTH//GRID_CELL_SIZE
    GRID_HEIGHT = HEIGHT//GRID_CELL_SIZE
    WRAP_AROUND = settings.WRAP_AROUND
    #######################################
    # Arrays
    #######################################
    global renderBuffer, boidBuffer, dirBuffer, lookUpTable, tileIndexTable, boidTable, tileOffsetTable
    renderBuffer = np.zeros((POPULATION, 2), dtype=np.float32)
    dirBuffer = cuda.to_device(np.zeros((POPULATION, 27), dtype=np.float32))
    tempBoidBuffer = np.zeros((POPULATION, 4), dtype=np.float32)
    for i in range(POPULATION):
        tempBoidBuffer[i,0] = rd.uniform(0, WIDTH)
        tempBoidBuffer[i,1] = rd.uniform(0, HEIGHT)
        tempBoidBuffer[i,2] = rd.random()*2-1
        tempBoidBuffer[i,3] = rd.random()*2-1
    boidBuffer = cuda.to_device(tempBoidBuffer)
    # Tables
    # Boid Table - List of every boid and their current tile, sorted by tile
    tempBoidTable = np.zeros((POPULATION,2), dtype=np.int32)
    for index, cell in enumerate(tempBoidTable):
        cell[0] = index
    # Sends array to GPU with CuPy instead of numba, to use CuPy methods
    boidTable = cp.asarray(tempBoidTable)
    # Tables
    # Tile Index Table (gives start index of tile on sorted Boid Table)
    tileIndexTable = np.zeros(GRID_WIDTH*GRID_HEIGHT, dtype=np.int32)
    # Tables
    # Look-Up table (for neighbor tiles - read only)
    tempLookUpTable = np.zeros((GRID_WIDTH*GRID_HEIGHT,9), dtype=np.int32)
    tileOffset = {-1, 0, 1}
    for index, cell in enumerate(tempLookUpTable):
        x = index % GRID_WIDTH
        y = index //GRID_WIDTH
        col = 0
        for i in tileOffset:
            for j in tileOffset:
                cell[col] = (x + i)%GRID_WIDTH + (y + j)%GRID_HEIGHT * GRID_WIDTH
                col += 1
    lookUpTable = cuda.to_device(tempLookUpTable)    

#######################################
# Update
#######################################
def update():
    global renderBuffer, dirBuffer, boidBuffer, lookUpTable, boidTable, tileIndexTable
    # Kernels are set to default stream and are executed sequentially
    # 512 threads per block, as many blocks as we need 
    nthreads = 512
    nblocks = (POPULATION + (nthreads - 1)) // nthreads
    fillBoidTable[nblocks, nthreads](boidBuffer, boidTable)
    # CuPy sort, a bit faster than on CPU
    boidTable = boidTable[cp.argsort(boidTable[:,1])]
    # New index table according to data from sorted boid table
    tileIndexTable.fill(-1)
    fillTileIndexTable[nblocks, nthreads](boidTable, tileIndexTable)
    # 2D kernel. X axis is boids, Y axis is each of their 9 respective neighbor tiles
    getBoidDataFromTile[(nblocks,9), (nthreads,1)](boidBuffer, dirBuffer, lookUpTable, tileIndexTable, boidTable)
    # Once previous kernel is finished, gets data from dirBuffer and writes to renderBuffer and boidBuffer
    writeToBuffer[nblocks, nthreads](boidBuffer, renderBuffer, dirBuffer)


#######################################
# Parallel
#######################################

# Update Boid Table with new tile for every agent
@cuda.jit
def fillBoidTable(boidBuffer, boidTable):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    x = int(boidBuffer[index,0]//GRID_CELL_SIZE) % GRID_WIDTH
    y = int(boidBuffer[index,1]//GRID_CELL_SIZE) % GRID_HEIGHT
    gridIndex = x + y * GRID_WIDTH
    boidTable[index,1] = gridIndex

# Update Index Table to give start index for every tile on tileIndexTable
@cuda.jit
def fillTileIndexTable(boidTable, tileIndexTable):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    gridIndex = boidTable[index,1]
    if index > 0:
        previousGridIndex = boidTable[index-1,1]
        if previousGridIndex == gridIndex:
            return
    tileIndexTable[gridIndex] = index

# For a certain agent and a certain tile in the agent's neighboring tiles
# Get neighbor data and compute new direction vector with weight
# Boidbuffer -> DirBuffer
@cuda.jit
def getBoidDataFromTile(boidBuffer, dirBuffer, lookUpTable, tileIndexTable, boidTable):
    lookUpIndex = cuda.blockIdx.y
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if index >= POPULATION:
        return
    # Current  position
    x, y = boidBuffer[index,0], boidBuffer[index,1]
    # Boid direction
    dx, dy = 0.0, 0.0
    numNeighbors = 0
    alignmentX, alignmentY = 0.0, 0.0
    cohesionX, cohesionY = 0.0, 0.0
    separationX, separationY = 0.0, 0.0
    numSeparation = 0
    #Check if close to edge (for distanche check)
    if WRAP_AROUND and (x < 100 or y < 100 or WIDTH - x < 100 or HEIGHT - y < 100):
        onEdge = True
    else: 
        onEdge = False
    # Get cell neighbors
    gridIndex = int(x//GRID_CELL_SIZE)%GRID_WIDTH + int(y//GRID_CELL_SIZE)%GRID_HEIGHT * GRID_WIDTH
    tile = lookUpTable[gridIndex, lookUpIndex]
    startIndex = tileIndexTable[tile]
    if startIndex != -1:
        value = tile
        while value == tile and startIndex < POPULATION:
            agent = boidTable[startIndex,0]
            startIndex += 1
            if startIndex == POPULATION:
                value = -1
            else:
                value = boidTable[startIndex,1]
            ax = boidBuffer[agent,0]
            ay = boidBuffer[agent,1]
            adx = boidBuffer[agent,2]
            ady = boidBuffer[agent,3]
            if onEdge:
                isNeighbor = isNeighborEdge(x,y,ax,ay)
            else: 
                isNeighbor = (((x - ax)**2 + (y - ay)**2) < NEIGHBOR_DIST_SQUARED)
            if agent != index and isNeighbor:
                numNeighbors += 1
                alignmentX += adx
                alignmentY += ady
                cohesionX += ax
                cohesionY += ay
                distX = x - ax
                distY = y - ay
                distLength = distX**2 + distY**2
                if distLength < SEPARATION_DIST_SQUARED:
                    distX /= distLength
                    distY /= distLength
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
        dx += alignmentX * ALIGNMENT
        dy += alignmentY * ALIGNMENT
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
        dx += cohesionX * COHESION
        dy += cohesionY * COHESION
        # Separation
        if numSeparation > 0:
            separationX /= numSeparation
            separationY /= numSeparation
            # Normalize separation vector
            l = math.sqrt(separationX**2 + separationY**2)
            separationX, separationY = separationX / l, separationY / l
        # Add separation vector to direction
        dx += separationX * SEPARATION
        dy += separationY * SEPARATION
    # Write to direction buffer at respective boid and tile index
    dirBuffer[index, lookUpIndex*3] = dx
    dirBuffer[index, lookUpIndex*3+1] = dy
    dirBuffer[index, lookUpIndex*3+2] = numNeighbors

# Reads direction vectors and weights for all neighbor tile calculations
# Gets final direction and position and writes to memory
# DirBuffer -> RenderBuffer, BoidBuffer
@cuda.jit
def writeToBuffer(boidBuffer, renderBuffer, dirBuffer):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    x, y = boidBuffer[index,0], boidBuffer[index,1]
    # Previous direction
    pdx, pdy = boidBuffer[index,2], boidBuffer[index,3]
    dx = 0.0
    dy = 0.0
    total = 0
    for i in range(9):
        weight = dirBuffer[index, i*3+2]
        total += weight
        dx += dirBuffer[index, i*3] * weight
        dy += dirBuffer[index, i*3+1] * weight
    if total > 0:
        dx /= total
        dy /= total
    # If Edge Wrapping is off, avoid walls
    MARGIN = 200
    if not WRAP_AROUND:
        wallX, wallY = 0.0, 0.0
        x2 = WIDTH - x 
        y2 = HEIGHT - y
        if x < MARGIN:
            wallX = MARGIN - x
            wallX /= MARGIN
            wallX = wallX ** 2
        elif x2 < MARGIN:
            wallX = MARGIN - x2
            wallX /= MARGIN
            wallX = wallX ** 2
            wallX *= -1
        if y < MARGIN:
            wallY = MARGIN - y
            wallY /= MARGIN
            wallY = wallY ** 2
        elif y2 < MARGIN:
            wallY = MARGIN - y2
            wallY /= MARGIN
            wallY = wallY ** 2
            wallY *= -1
        dx += wallX /10
        dy += wallY /10
    # Update directions
    dx = pdx + dx * 3 # Weight w, ratio of 1:w with old/new direction
    dy = pdy + dy * 3
    # Clamp speed
    l = math.sqrt(dx**2 + dy**2)
    dx, dy = dx / l, dy / l
    # Move position
    x += dx * SPEED
    y += dy * SPEED
    # Edge wrap
    if WRAP_AROUND:
        if x > WIDTH:
            x = 0
        elif x < 0 :
            x = WIDTH
        if y > HEIGHT:
            y = 0
        elif y < 0 :
            y = HEIGHT
    # Render coordinates
    rx = (x - HALF_WIDTH) / HALF_WIDTH
    ry = (y - HALF_HEIGHT) / HALF_HEIGHT
    # WRITE TO MEMORY
    # Write screen position in [-1,1]
    renderBuffer[index,0] = rx
    renderBuffer[index,1] = ry
    # Store data for next calculation
    boidBuffer[index,0] = x
    boidBuffer[index,1] = y
    boidBuffer[index,2] = dx
    boidBuffer[index,3] = dy

# If agent is near an edge
# Checks if other agent is a neighbor for all neighboring tile configurations (9)
# Looks for minimal toroidal distance
@cuda.jit(device=True)
def isNeighborEdge(x,y,ax,ay):
    mix = min(abs(x - ax - WIDTH), abs(x - ax), abs(x - ax + WIDTH))
    miy = min(abs(y - ay - HEIGHT), abs(y - ay), abs(y - ay + HEIGHT))
    if ((mix)**2 + (miy)**2) < NEIGHBOR_DIST_SQUARED:
        return True
    return False


from numba import cuda, float32
import numpy as np
import cupy as cp
import random as rd
import math

def init(settings):
    #Validation
    if not cuda.is_available():
        print("No CUDA device found.")
        exit()
    #######################################
    # Constants
    #######################################
    global POPULATION, SPEED, WRAP_AROUND, SPOTLIGHT
    global COHESION, ALIGNMENT, SEPARATION, NEIGHBOR_DIST, NEIGHBOR_DIST_SQUARED, SEPARATION_DIST_SQUARED
    global WIDTH, HEIGHT, HALF_WIDTH, HALF_HEIGHT, SPEED, PARALLEL, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, GRID_SIZE
    POPULATION = settings.POPULATION
    SPEED = settings.SPEED
    COHESION = settings.COHESION
    ALIGNMENT = settings.ALIGNMENT
    SEPARATION = settings.SEPARATION
    # Constants used in kernel float math are float32, otherwise Numba promotes that math to float64
    NEIGHBOR_DIST = np.float32(settings.NEIGHBOR_DIST)
    NEIGHBOR_DIST_SQUARED = np.float32(settings.NEIGHBOR_DIST ** 2) # Useful for evaluating distances
    SEPARATION_DIST_SQUARED = (settings.NEIGHBOR_DIST * settings.SEPARATION_DIST) ** 2
    WIDTH = np.float32(settings.WIDTH)
    HEIGHT = np.float32(settings.HEIGHT)
    HALF_WIDTH = np.float32(settings.WIDTH/2)
    HALF_HEIGHT = np.float32(settings.HEIGHT/2)
    GRID_CELL_SIZE = np.float32(settings.NEIGHBOR_DIST)
    # Grid dimensions stay ints (array sizes, modulo, indexing)
    GRID_WIDTH = settings.WIDTH//settings.NEIGHBOR_DIST
    GRID_HEIGHT = settings.HEIGHT//settings.NEIGHBOR_DIST
    GRID_SIZE = GRID_WIDTH*GRID_HEIGHT
    WRAP_AROUND = settings.WRAP_AROUND
    SPOTLIGHT = settings.SPOTLIGHT
    #######################################
    # Arrays
    #######################################
    global boids, sortedBoids, partials, lookUpTable, cellIndexTable, boidPositionTable, unsortedBoidPositionTable
    # Boids - (x, y, dx, dy) for every agent, in agent order
    tempBoids = np.zeros((POPULATION, 4), dtype=np.float32)
    for i in range(POPULATION):
        tempBoids[i,0] = rd.uniform(0, WIDTH)
        tempBoids[i,1] = rd.uniform(0, HEIGHT)
        tempBoids[i,2] = rd.random()*2-1
        tempBoids[i,3] = rd.random()*2-1
    boids = cp.asarray(tempBoids)
    # Sorted Boids - copy of Boids in cell order, rebuilt every frame so that
    # agents of the same cell are contiguous in memory for the neighbor search
    sortedBoids = cp.empty_like(boids)
    # Partials - (x, y, weight) direction for every neighbor cell of every agent, indexed by sorted slot
    partials = cp.zeros((9, POPULATION, 3), dtype=cp.float32)
    # Tables
    # Boid Table - List of every boid and their current cell, sorted by cell
    # (filled unsorted, then sorted into a second table so both keep their memory)
    unsortedBoidPositionTable = cp.zeros((POPULATION,2), dtype=cp.int32)
    boidPositionTable = cp.zeros((POPULATION,2), dtype=cp.int32)
    # Tables
    # Cell Index Table (gives start index of cell on sorted Boid Table)
    cellIndexTable = cp.zeros(GRID_SIZE, dtype=cp.int32)
    # Tables
    # Look-Up table (for neighbor cells - read only)
    tempLookUpTable = np.zeros((GRID_SIZE,9), dtype=np.int32)
    cellOffset = {-1, 0, 1}
    for index, cell in enumerate(tempLookUpTable):
        x = index % GRID_WIDTH
        y = index //GRID_WIDTH
        col = 0
        for i in cellOffset:
            for j in cellOffset:
                cell[col] = (x + i)%GRID_WIDTH + (y + j)%GRID_HEIGHT * GRID_WIDTH
                col += 1
    lookUpTable = cuda.to_device(tempLookUpTable)
    # Numba views of the CuPy arrays, made once: converting CuPy arrays on every launch is slow
    global nbBoids, nbSortedBoids, nbPartials, nbCellIndexTable, nbBoidPositionTable, nbUnsortedBoidPositionTable
    nbBoids = cuda.as_cuda_array(boids)
    nbSortedBoids = cuda.as_cuda_array(sortedBoids)
    nbPartials = cuda.as_cuda_array(partials)
    nbCellIndexTable = cuda.as_cuda_array(cellIndexTable)
    nbBoidPositionTable = cuda.as_cuda_array(boidPositionTable)
    nbUnsortedBoidPositionTable = cuda.as_cuda_array(unsortedBoidPositionTable)
    #######################################
    # Kernels
    #######################################
    # Compiled here, as they read the constants above. Fixed signatures: other scalar types are
    # converted to these, arrays of other types raise, and no slower (e.g. float64) version is compiled
    compileKernel(fillBoidPositionTable, "void(float32[:,::1], int32[:,::1], float32[:,::1])")
    compileKernel(fillCellIndexTable, "void(int32[:,::1], int32[::1])")
    compileKernel(neighborSearch, "void(float32[:,::1], float32[:,:,::1], int32[:,::1], int32[::1], int32[:,::1], float32[:,::1], "
        "boolean, boolean, float32, float32, float32, float32)")
    compileKernel(writeData, "void(float32[:,::1], float32[:,::1], float32[:,:,::1], int32[:,::1], float32[:,::1], boolean, float32)")
    #######################################
    # Stream and CUDA graph
    #######################################
    # All simulation work runs on one non-blocking stream, so a frame can be recorded
    # as a CUDA graph (the default stream can't be recorded)
    global stream, nbStream, graph, graphKey, graphPool
    stream = cp.cuda.Stream(non_blocking=True)
    nbStream = cuda.external_stream(stream.ptr)
    graph, graphKey = None, None
    # Temporary arrays of a recorded frame (the sort) come from this pool, which nothing
    # else uses, so their memory is never handed to other arrays while the graph uses it
    graphPool = cp.cuda.MemoryPool()
    # The arrays above were filled on the default stream
    cp.cuda.Device().synchronize()

def compileKernel(kernel, signature):
    kernel.compile(signature)
    kernel.disable_compile()

#######################################
# Update
#######################################

# renderData - (x, y, color) for every agent, the mapped OpenGL vertex buffer
def update(params, renderData):
    global graph, graphKey
    updateParams(params)
    # Kernel arguments and array addresses are recorded in the graph,
    # so it is recorded again when a parameter or the buffer address changes
    key = (SPEED, COHESION, ALIGNMENT, SEPARATION, SEPARATION_DIST_SQUARED, WRAP_AROUND, SPOTLIGHT, renderData.data.ptr)
    if key != graphKey:
        nbRenderData = cuda.as_cuda_array(renderData, sync=False)
        with stream, cp.cuda.using_allocator(graphPool.malloc):
            stream.begin_capture()
            try:
                recordFrame(nbRenderData)
            finally:
                graph = stream.end_capture()
        graphKey = key
    graph.launch(stream)

# Launches one simulation step on the stream (recorded into the graph, not executed)
def recordFrame(renderData):
    # 512 threads per block, as many blocks as we need
    nthreads = 512
    nblocks = (POPULATION + (nthreads - 1)) // nthreads
    fillBoidPositionTable[nblocks, nthreads, nbStream](nbBoids, nbUnsortedBoidPositionTable, renderData)
    # CuPy sort, a bit faster than on CPU
    order = cp.argsort(unsortedBoidPositionTable[:,1])
    cp.take(unsortedBoidPositionTable, order, axis=0, out=boidPositionTable)
    # Reorder agent data to match the sorted table
    cp.take(boids, order, axis=0, out=sortedBoids)
    # New index table according to data from sorted boid table
    cellIndexTable.fill(-1)
    fillCellIndexTable[nblocks, nthreads, nbStream](nbBoidPositionTable, nbCellIndexTable)
    # 2D kernel. X axis is sorted boids, Y axis is each of their 9 respective neighbor cells
    neighborSearch[(nblocks,9), (nthreads,1), nbStream](nbSortedBoids, nbPartials, lookUpTable, nbCellIndexTable, nbBoidPositionTable, renderData,
        SPOTLIGHT, WRAP_AROUND, COHESION, ALIGNMENT, SEPARATION, SEPARATION_DIST_SQUARED)
    # Update positions and write to render buffer
    writeData[nblocks, nthreads, nbStream](nbBoids, nbSortedBoids, nbPartials, nbBoidPositionTable, renderData, WRAP_AROUND, SPEED)

# Update parameters with GUI input
def updateParams(params):
    global POPULATION, SPEED, WRAP_AROUND, SPOTLIGHT
    global COHESION, ALIGNMENT, SEPARATION, NEIGHBOR_DIST, NEIGHBOR_DIST_SQUARED, SEPARATION_DIST_SQUARED
    POPULATION = params.POPULATION
    # Kernel arguments are float32, Python floats would arrive as float64
    SPEED = np.float32(params.SPEED)
    COHESION = np.float32(params.COHESION)
    ALIGNMENT = np.float32(params.ALIGNMENT)
    SEPARATION = np.float32(params.SEPARATION)
    SEPARATION_DIST_SQUARED = np.float32((params.NEIGHBOR_DIST * params.SEPARATION_DIST) ** 2)
    WRAP_AROUND = params.WRAP_AROUND
    SPOTLIGHT = params.SPOTLIGHT


#######################################
# Kernels
#######################################

# Update Boid Table with new cell for every agent
@cuda.jit(fastmath=True)
def fillBoidPositionTable(boids, boidPositionTable, renderData):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    x = int(boids[index,0]//GRID_CELL_SIZE) % GRID_WIDTH
    y = int(boids[index,1]//GRID_CELL_SIZE) % GRID_HEIGHT
    gridIndex = x + y * GRID_WIDTH
    boidPositionTable[index,0] = index
    boidPositionTable[index,1] = gridIndex
    renderData[index, 2] = 0.0


# Update Index Table to give start index for every cell on boidPositionTable
@cuda.jit(fastmath=True)
def fillCellIndexTable(boidPositionTable, cellIndexTable):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    gridIndex = boidPositionTable[index,1]
    if index > 0:
        previousGridIndex = boidPositionTable[index-1,1]
        if previousGridIndex == gridIndex:
            return
    cellIndexTable[gridIndex] = index

# For a certain agent and a certain cell in the agent's neighboring cells
# Get neighbor data and compute new direction vector with weight
# Threads work on sorted slots, so neighboring threads usually share a cell
# sortedBoids -> partials
@cuda.jit(fastmath=True)
def neighborSearch(sortedBoids, partials, lookUpTable, cellIndexTable, boidPositionTable, renderData, SPOTLIGHT, WRAP_AROUND, COHESION, ALIGNMENT, SEPARATION, SEPARATION_DIST_SQUARED):
    lookUpIndex = cuda.blockIdx.y
    slot = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if slot >= POPULATION:
        return
    # Agent id, only needed for the spotlight
    index = boidPositionTable[slot,0]
    # Current  position
    x, y = sortedBoids[slot,0], sortedBoids[slot,1]
    dx, dy = sortedBoids[slot,2], sortedBoids[slot,3]
    # Boid direction
    resX, resY = float32(0.0), float32(0.0)
    numNeighbors = 0
    alignmentX, alignmentY = float32(0.0), float32(0.0)
    cohesionX, cohesionY = float32(0.0), float32(0.0)
    separationX, separationY = float32(0.0), float32(0.0)
    numSeparation = 0
    #Check if close to edge (for distanche check)
    if WRAP_AROUND and (x < NEIGHBOR_DIST or y < NEIGHBOR_DIST or WIDTH - x < NEIGHBOR_DIST or HEIGHT - y < NEIGHBOR_DIST):
        onEdge = True
    else:
        onEdge = False
    # Get cell neighbors (the agent's own cell is already in the sorted table)
    gridIndex = boidPositionTable[slot,1]
    cell = lookUpTable[gridIndex, lookUpIndex]
    startIndex = cellIndexTable[cell]
    if startIndex != -1: # True if there's at least one agent on the current cell
        # Condition to check if there are more agents to observe on the cell
        cond = cell
        while cond == cell and startIndex < POPULATION:# and numNeighbors < 20:
            other = startIndex
            startIndex += 1
            # Check if at the end of table
            if startIndex >= POPULATION:
                cond = -1
            # Or set condition to check if all agents on the cell have been observed
            else:
                cond = boidPositionTable[startIndex,1]
            # Get agent data and check if it's a neighbor
            if other != slot:
                ax = sortedBoids[other,0]
                ay = sortedBoids[other,1]
                adx = sortedBoids[other,2]
                ady = sortedBoids[other,3]
                if onEdge:
                    (ax, ay) = minimalToroidalDistance(x,y,ax,ay)
                distX = x - ax
                distY = y - ay
                distLengthSquared = distX**2 + distY**2
                isTooClose = (distLengthSquared < SEPARATION_DIST_SQUARED)
                isNeighbor = (distLengthSquared < NEIGHBOR_DIST_SQUARED) and isInFOV(dx,dy,distX,distY,distLengthSquared)
                # SPOTLIGHT
                if SPOTLIGHT and index == 0:
                    # Agent spotlight
                    agent = boidPositionTable[other,0]
                    renderData[index, 2] = 1.0
                    if isTooClose:
                        renderData[agent, 2] = 2.0
                    elif isNeighbor:
                        renderData[agent, 2] = 3.0
                    else:
                        renderData[agent, 2] = 4.0
                # Collect agent data
                if isNeighbor:
                    numNeighbors += 1
                    alignmentX += adx
                    alignmentY += ady
                    cohesionX += ax
                    cohesionY += ay
                    if isTooClose:
                        # distX /= distLengthSquared
                        # distY /= distLengthSquared
                        separationX += distX
                        separationY += distY
                        numSeparation += 1
    if numNeighbors > 0:
        # Apply boid rules
        # Alignment
        # Normalize alignment vector
        l = math.sqrt(alignmentX**2 + alignmentY**2)
        alignmentX, alignmentY = alignmentX / l, alignmentY / l
        # Add alignment vector to direction
        resX += alignmentX * ALIGNMENT
        resY += alignmentY * ALIGNMENT
        # Cohesion
        # Get vector towards center of mass
        cohesionX /= float32(numNeighbors)
        cohesionY /= float32(numNeighbors)
        cohesionX -= x
        cohesionY -= y
        # Normalize cohesion vector
        l = math.sqrt(cohesionX**2 + cohesionY**2)
        cohesionX, cohesionY = cohesionX / l, cohesionY / l
        # Add cohesion vector to direction
        resX += cohesionX * COHESION
        resY += cohesionY * COHESION
        # Separation
        if numSeparation > 0:
            # Normalize separation vector
            l = math.sqrt(separationX**2 + separationY**2)
            separationX, separationY = separationX / l, separationY / l
            # Add separation vector to direction
            resX += separationX * SEPARATION
            resY += separationY * SEPARATION
    # Write to partials buffer at respective cell and sorted slot
    partials[lookUpIndex, slot, 0] = resX
    partials[lookUpIndex, slot, 1] = resY
    partials[lookUpIndex, slot, 2] = numNeighbors

# Reads direction vectors and weights for all neighbor cell calculations
# Gets final direction and position and writes to memory (back in agent order)
# sortedBoids, partials -> boids, renderData
@cuda.jit(fastmath=True)
def writeData(boids, sortedBoids, partials, boidPositionTable, renderData, WRAP_AROUND, SPEED):
    slot = cuda.grid(1)
    if slot >= POPULATION:
        return
    index = boidPositionTable[slot,0]
    x, y = sortedBoids[slot,0], sortedBoids[slot,1]
    # Previous direction
    pdx, pdy = sortedBoids[slot,2], sortedBoids[slot,3]
    dx = float32(0.0)
    dy = float32(0.0)
    total = float32(0.0)
    for i in range(9):
        weight = partials[i, slot, 2]
        total += weight
        dx += partials[i, slot, 0] * weight
        dy += partials[i, slot, 1] * weight
    if total > float32(0.0):
        dx /= total
        dy /= total
    # If Edge Wrapping is off, avoid walls
    MARGIN = float32(200.0)
    if not WRAP_AROUND:
        wallX, wallY = float32(0.0), float32(0.0)
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
            wallX = -wallX
        if y < MARGIN:
            wallY = MARGIN - y
            wallY /= MARGIN
            wallY = wallY ** 2
        elif y2 < MARGIN:
            wallY = MARGIN - y2
            wallY /= MARGIN
            wallY = wallY ** 2
            wallY = -wallY
        dx += wallX / float32(10.0)
        dy += wallY / float32(10.0)
    # Update directions
    dx = pdx + dx * float32(3.0) # Weight w, ratio of 1:w with old/new direction
    dy = pdy + dy * float32(3.0)
    # Clamp speed
    l = math.sqrt(dx**2 + dy**2)
    if l > float32(1.0):
        dx, dy = dx / l, dy / l
    # Move position
    x += dx * SPEED
    y += dy * SPEED
    # Edge wrap
    if WRAP_AROUND:
        if x >= WIDTH:
            x = float32(0.0)
        elif x < float32(0.0):
            x = WIDTH
        if y >= HEIGHT:
            y = float32(0.0)
        elif y < float32(0.0):
            y = HEIGHT
    # Render coordinates
    rx = (x - HALF_WIDTH) / HALF_WIDTH
    ry = (y - HALF_HEIGHT) / HALF_HEIGHT
    # WRITE TO MEMORY
    # Write screen position in [-1,1]
    renderData[index,0] = rx
    renderData[index,1] = ry
    # renderData[index,2] = ry
    # Store data for next calculation
    boids[index,0] = x
    boids[index,1] = y
    boids[index,2] = dx
    boids[index,3] = dy

# If agent is near an edge and WrapAround is set to True
# Checks if other agent is a neighbor for all neighboring cell configurations (9)
# Looks for minimal toroidal distance and returns perceived position
# Given what is provided by Numba we have to get the minimum manually
@cuda.jit(device=True, fastmath=True)
def minimalToroidalDistance(x,y,ax,ay):
    # X axis
    ax1 = ax - WIDTH
    ax2 = ax + WIDTH
    mix = abs(x - ax)
    mix1 = abs(x - ax1)
    mix2 = abs(x - ax2)
    if mix1 < mix:
        mix = mix1
        ax = ax1
    if mix2 < mix:
        mix = mix2
        ax = ax2
    # Y axis
    ay1 = ay - HEIGHT
    ay2 = ay + HEIGHT
    miy = abs(y - ay)
    miy1 = abs(y - ay1)
    miy2 = abs(y - ay2)
    if miy1 < miy:
        miy = miy1
        ay = ay1
    if miy2 < miy:
        miy = miy2
        ay = ay2
    return (ax,ay)

@cuda.jit(device=True, fastmath=True)
def isInFOV(dx,dy,adx,ady,neighborVector):
    return ( (dx * adx + dy * ady) / (math.sqrt(dx**2 + dy**2) * math.sqrt(neighborVector)) ) < float32(0.8)
    # ?
    # return ( (dx * adx + dy * ady)**2 / ((dx**2 + dy**2) * neighborVector) ) < 0.64

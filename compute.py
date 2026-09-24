from numba import cuda, float32, int32
import numpy as np
import cupy as cp
import random as rd
import math

# Threads per block of the neighbor search, and candidates loaded in shared memory at a time
TILE_SIZE = 128
# Blocks per cell: each takes every CELL_BLOCKS-th group of TILE_SIZE agents of the cell
CELL_BLOCKS = 8

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
    global boids, sortedBoids, steering, frameCounter, lookUpTable, cellIndexTable, cellEndTable, boidPositionTable, unsortedBoidPositionTable
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
    # Steering - (x, y) direction given by the three rules for every agent, indexed by sorted slot
    steering = cp.zeros((POPULATION, 2), dtype=cp.float32)
    # Frame counter, seeds the random sampling of candidates (kernel arguments are fixed in the graph)
    frameCounter = cp.zeros(1, dtype=cp.int32)
    # Tables
    # Boid Table - List of every boid and their current cell, sorted by cell
    # (filled unsorted, then sorted into a second table so both keep their memory)
    unsortedBoidPositionTable = cp.zeros((POPULATION,2), dtype=cp.int32)
    boidPositionTable = cp.zeros((POPULATION,2), dtype=cp.int32)
    # Tables
    # Cell Index Tables (give start and end index of cell on sorted Boid Table)
    cellIndexTable = cp.zeros(GRID_SIZE, dtype=cp.int32)
    cellEndTable = cp.zeros(GRID_SIZE, dtype=cp.int32)
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
    global nbBoids, nbSortedBoids, nbSteering, nbFrameCounter, nbCellIndexTable, nbCellEndTable, nbBoidPositionTable, nbUnsortedBoidPositionTable
    nbBoids = cuda.as_cuda_array(boids)
    nbSortedBoids = cuda.as_cuda_array(sortedBoids)
    nbSteering = cuda.as_cuda_array(steering)
    nbFrameCounter = cuda.as_cuda_array(frameCounter)
    nbCellIndexTable = cuda.as_cuda_array(cellIndexTable)
    nbCellEndTable = cuda.as_cuda_array(cellEndTable)
    nbBoidPositionTable = cuda.as_cuda_array(boidPositionTable)
    nbUnsortedBoidPositionTable = cuda.as_cuda_array(unsortedBoidPositionTable)
    #######################################
    # Kernels
    #######################################
    # Compiled here, as they read the constants above. Fixed signatures: other scalar types are
    # converted to these, arrays of other types raise, and no slower (e.g. float64) version is compiled
    compileKernel(fillBoidPositionTable, "void(float32[:,::1], int32[:,::1], float32[:,::1])")
    compileKernel(fillCellIndexTable, "void(int32[:,::1], int32[::1], int32[::1])")
    compileKernel(neighborSearch, "void(float32[:,::1], float32[:,::1], int32[:,::1], int32[::1], int32[::1], int32[:,::1], float32[:,::1], int32[::1], "
        "boolean, boolean, float32, float32, float32, float32, int32)")
    compileKernel(writeData, "void(float32[:,::1], float32[:,::1], float32[:,::1], int32[:,::1], float32[:,::1], boolean, float32)")
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
    key = (SPEED, COHESION, ALIGNMENT, SEPARATION, SEPARATION_DIST_SQUARED, WRAP_AROUND, SPOTLIGHT, CANDIDATE_BUDGET, renderData.data.ptr)
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
    cp.add(frameCounter, 1, out=frameCounter)
    fillBoidPositionTable[nblocks, nthreads, nbStream](nbBoids, nbUnsortedBoidPositionTable, renderData)
    # CuPy sort, a bit faster than on CPU
    order = cp.argsort(unsortedBoidPositionTable[:,1])
    cp.take(unsortedBoidPositionTable, order, axis=0, out=boidPositionTable)
    # Reorder agent data to match the sorted table
    cp.take(boids, order, axis=0, out=sortedBoids)
    # New index tables according to data from sorted boid table (empty cells stay at start = end = 0)
    cellIndexTable.fill(0)
    cellEndTable.fill(0)
    fillCellIndexTable[nblocks, nthreads, nbStream](nbBoidPositionTable, nbCellIndexTable, nbCellEndTable)
    # CELL_BLOCKS blocks per cell
    neighborSearch[(GRID_SIZE, CELL_BLOCKS), TILE_SIZE, nbStream](nbSortedBoids, nbSteering, lookUpTable, nbCellIndexTable, nbCellEndTable, nbBoidPositionTable, renderData,
        nbFrameCounter, SPOTLIGHT, WRAP_AROUND, COHESION, ALIGNMENT, SEPARATION, SEPARATION_DIST_SQUARED, CANDIDATE_BUDGET)
    # Update positions and write to render buffer
    writeData[nblocks, nthreads, nbStream](nbBoids, nbSortedBoids, nbSteering, nbBoidPositionTable, renderData, WRAP_AROUND, SPEED)

# Update parameters with GUI input
def updateParams(params):
    global POPULATION, SPEED, WRAP_AROUND, SPOTLIGHT, CANDIDATE_BUDGET
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
    CANDIDATE_BUDGET = np.int32(max(1, params.CANDIDATE_BUDGET))


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


# Update Index Tables to give start and end (exclusive) index for every cell on boidPositionTable
@cuda.jit(fastmath=True)
def fillCellIndexTable(boidPositionTable, cellIndexTable, cellEndTable):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    gridIndex = boidPositionTable[index,1]
    # First agent of its cell
    if index == 0 or boidPositionTable[index-1,1] != gridIndex:
        cellIndexTable[gridIndex] = index
    # Last agent of its cell
    if index == POPULATION-1 or boidPositionTable[index+1,1] != gridIndex:
        cellEndTable[gridIndex] = index + 1

# One block per cell: its threads take the agents of the cell (TILE_SIZE at a time) and
# go through the agents of the 9 neighbor cells together, seen as one list of candidates.
# Those are loaded in shared memory one tile at a time, and every thread tests its own agent
# against the tile. All threads of a block read the same candidate at the same time and loop
# the same number of times. The three rules are then applied to all of an agent's neighbors at once.
# When there are more than CANDIDATE_BUDGET candidates, only every stride-th one is tested,
# from a random offset that changes every frame: every candidate has the same chance to be
# tested whatever its cell or position, so the sample isn't biased in any direction
# sortedBoids -> steering
@cuda.jit(fastmath=True)
def neighborSearch(sortedBoids, steering, lookUpTable, cellIndexTable, cellEndTable, boidPositionTable, renderData, frameCounter, SPOTLIGHT, WRAP_AROUND, COHESION, ALIGNMENT, SEPARATION, SEPARATION_DIST_SQUARED, CANDIDATE_BUDGET):
    cell = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    cellStart, cellEnd = cellIndexTable[cell], cellEndTable[cell]
    # Blocks without agents of their own (the whole block returns together)
    if cellStart + cuda.blockIdx.y * TILE_SIZE >= cellEnd:
        return
    # Candidate list: the agents of the 9 neighbor cells, one after the other.
    # listStart[k] is where cell k starts in the list, cellFirst[k] where it starts in sortedBoids
    listStart = cuda.shared.array(10, int32)
    cellFirst = cuda.shared.array(9, int32)
    if thread == 0:
        listStart[0] = 0
        for k in range(9):
            neighborCell = lookUpTable[cell, k]
            cellFirst[k] = cellIndexTable[neighborCell]
            listStart[k+1] = listStart[k] + cellEndTable[neighborCell] - cellIndexTable[neighborCell]
    cuda.syncthreads()
    numCandidates = listStart[9]
    # Sample every stride-th candidate from a random offset when over the budget
    stride = 1
    offset = 0
    if numCandidates > CANDIDATE_BUDGET:
        stride = (numCandidates + CANDIDATE_BUDGET - 1) // CANDIDATE_BUDGET
        offset = randomHash(cell, cuda.blockIdx.y, frameCounter[0]) % stride
    numSampled = (numCandidates - offset + stride - 1) // stride
    # Candidates (x, y, dx, dy) and their slots, shared by the threads of the block
    tile = cuda.shared.array((TILE_SIZE, 4), float32)
    tileSlot = cuda.shared.array(TILE_SIZE, int32)
    # Agents of the cell, TILE_SIZE at a time, shared between the blocks of the cell
    # (the whole block skips the loop together when it has no agents)
    for first in range(cellStart + cuda.blockIdx.y * TILE_SIZE, cellEnd, TILE_SIZE * CELL_BLOCKS):
        slot = first + thread
        # Threads past the end of the cell still help loading tiles
        active = slot < cellEnd
        index = -1
        x, y, dx, dy = float32(0.0), float32(0.0), float32(0.0), float32(0.0)
        if active:
            # Agent id, only needed for the spotlight
            index = boidPositionTable[slot,0]
            # Current  position
            x, y = sortedBoids[slot,0], sortedBoids[slot,1]
            dx, dy = sortedBoids[slot,2], sortedBoids[slot,3]
        # Squared length of the direction, for the field of view test
        directionSquared = dx**2 + dy**2
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
        for tileStart in range(0, numSampled, TILE_SIZE):
            # Load a tile of candidates, one per thread
            i = tileStart + thread
            if i < numSampled:
                # Position in the candidate list, then in sortedBoids
                m = offset + i * stride
                k = 0
                while listStart[k+1] <= m:
                    k += 1
                candidate = cellFirst[k] + m - listStart[k]
                tileSlot[thread] = candidate
                for j in range(4):
                    tile[thread, j] = sortedBoids[candidate, j]
            cuda.syncthreads()
            if active:
                for t in range(min(TILE_SIZE, numSampled - tileStart)):
                    other = tileSlot[t]
                    # Get agent data and check if it's a neighbor
                    if other != slot:
                        ax = tile[t,0]
                        ay = tile[t,1]
                        adx = tile[t,2]
                        ady = tile[t,3]
                        if onEdge:
                            (ax, ay) = minimalToroidalDistance(x,y,ax,ay)
                        distX = x - ax
                        distY = y - ay
                        distLengthSquared = distX**2 + distY**2
                        isTooClose = (distLengthSquared < SEPARATION_DIST_SQUARED)
                        isNeighbor = (distLengthSquared < NEIGHBOR_DIST_SQUARED) and isInFOV(dx,dy,distX,distY,directionSquared,distLengthSquared)
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
            # The tile is overwritten next
            cuda.syncthreads()
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
        # Write to steering buffer at sorted slot
        if active:
            steering[slot, 0] = resX
            steering[slot, 1] = resY

# Reads the steering direction given by the rules
# Gets final direction and position and writes to memory (back in agent order)
# sortedBoids, steering -> boids, renderData
@cuda.jit(fastmath=True)
def writeData(boids, sortedBoids, steering, boidPositionTable, renderData, WRAP_AROUND, SPEED):
    slot = cuda.grid(1)
    if slot >= POPULATION:
        return
    index = boidPositionTable[slot,0]
    x, y = sortedBoids[slot,0], sortedBoids[slot,1]
    # Previous direction
    pdx, pdy = sortedBoids[slot,2], sortedBoids[slot,3]
    dx, dy = steering[slot,0], steering[slot,1]
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

# True if the other agent is outside the blind spot, i.e. cos(angle between the direction and
# the vector from the other agent) < 0.8. Without sqrt or division: when the dot product is
# negative the cosine is too, otherwise both sides can be squared
@cuda.jit(device=True, fastmath=True)
def isInFOV(dx,dy,distX,distY,directionSquared,distLengthSquared):
    dot = dx * distX + dy * distY
    return dot < float32(0.0) or dot**2 < float32(0.64) * directionSquared * distLengthSquared

# Integer hash ("lowbias32"), gives a well mixed non-negative number from three integers
@cuda.jit(device=True)
def randomHash(a, b, c):
    h = (a * 73856093 + b * 19349663 + c * 83492791) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x7feb352d) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x846ca68b) & 0xFFFFFFFF
    h ^= h >> 16
    return h

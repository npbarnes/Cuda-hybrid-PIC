__device__ int getBlockId()
{
    return blockIdx.x 
        + blockIdx.y * gridDim.x 
        + gridDim.x * gridDim.y * blockIdx.z;
}

__device__ int getThreadId()
{
    return (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
}

__device__ int getGlobalId()
{
    int blockId = getBlockId();
    int threadId = getThreadId();

    return blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId;
}

__device__ int index(int partid, int comp)
{
    return 3*partid + comp;
}

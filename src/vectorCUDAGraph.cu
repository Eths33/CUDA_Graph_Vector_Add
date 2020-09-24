/*
* Author: Gregory Gutmann
* Simple demonstration of CUDA graphs using the vector add code from Visual Studio's default CUDA project as a starting point.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include <vector>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

#define LOOP_COUNT 100
#define VERBOSE 0

typedef std::chrono::high_resolution_clock Clock;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, int loopCount, __int64 *tElapsed);
cudaError_t addWithCudaGraph(int *c, const int *a, const int *b, unsigned int size, int loopCount, __int64 * tElapsedGraph);

int originalTest();
int extendedTest();

__global__ void addKernel(int *c, const int *a, const int *b, int n)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    c[gtid] = a[gtid] + b[gtid];
}

int main()
{
    if (originalTest()) {
        fprintf(stderr, "originalTest failed!");
        return 1;
    }

    if (extendedTest()) {
        fprintf(stderr, "extendedTest failed!");
        return 1;
    }

    // Print warning, assumes code ran successfully
    printf("\nWARNING: If loop count is low these timings may be skewed by GPU warmup time\n");

    return 0;
}

int originalTest() {
    const int arraySize = 5;

    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    __int64 tElapsed = 0;
    __int64 tElapsedGraph = 0;

    // Add vectors in parallel.
    printf("\nRunning addWithCuda: arraySize = %d\n", arraySize);

    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, LOOP_COUNT, &tElapsed);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	if (VERBOSE)
		printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
			c[0], c[1], c[2], c[3], c[4]);

    // Add vectors in parallel with graph.
    printf("\nRunning addWithCudaGraph: arraySize = %d\n", arraySize);

    cudaStatus = addWithCudaGraph(c, a, b, arraySize, LOOP_COUNT, &tElapsedGraph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCudaGraph failed!");
        return 1;
    }

	if (VERBOSE)
		printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
			c[0], c[1], c[2], c[3], c[4]);

    printf("\n<Using a graph %s time by %lld microseconds a %3.2fx gain>\n",
        (tElapsed > tElapsedGraph) ? "reduced" : "increased", 
        abs(tElapsed - tElapsedGraph), tElapsed / (double)tElapsedGraph);

    printf("\n----------------------------------------------------------------------------\n");
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

int extendedTest() {
    const int arraySize = 1 << 18;

    int *a;
    int *b;
    int *c;
    
    __int64 tElapsed = 0;
    __int64 tElapsedGraph = 0;

    // Greater Host <-> Device memory copy performance (normally use sparingly)
    cudaMallocHost((void**)&a, arraySize * sizeof(int));
    cudaMallocHost((void**)&b, arraySize * sizeof(int));
    cudaMallocHost((void**)&c, arraySize * sizeof(int));

    int i = 0;
#pragma omp parallel for
    for (i = 0; i < arraySize; ++i) {
        a[i] = rand() % 20;
        b[i] = rand() % 20;
        c[i] = 0;
    }

    // Add vectors in parallel.
    printf("\nRunning addWithCuda: arraySize = %d\n", arraySize);
    
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, LOOP_COUNT, &tElapsed);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	if (VERBOSE)
		printf("{%d,%d,%d,%d,%d,...} + {%d,%d,%d,%d,%d,...} = {%d,%d,%d,%d,%d,...}\n",
			a[0], a[1], a[2], a[3], a[4],
			b[0], b[1], b[2], b[3], b[4],
			c[0], c[1], c[2], c[3], c[4]);

    // Add vectors in parallel with graph.
    printf("\nRunning addWithCudaGraph: arraySize = %d\n", arraySize);
    
    cudaStatus = addWithCudaGraph(c, a, b, arraySize, LOOP_COUNT, &tElapsedGraph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCudaGraph failed!");
        return 1;
    }

	if (VERBOSE)
		printf("{%d,%d,%d,%d,%d,...} + {%d,%d,%d,%d,%d,...} = {%d,%d,%d,%d,%d,...}\n",
			a[0], a[1], a[2], a[3], a[4],
			b[0], b[1], b[2], b[3], b[4],
			c[0], c[1], c[2], c[3], c[4]);

    printf("\n<Using a graph %s time by %lld microseconds a %3.2fx gain>\n", 
        (tElapsed > tElapsedGraph) ? "reduced" : "increased", 
        abs(tElapsed - tElapsedGraph), tElapsed / (double)tElapsedGraph);

    // Free CUDA host memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("\n----------------------------------------------------------------------------\n");
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, int loopCount, __int64* tElapsed)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Pre-declare timers for reducing warnings related to the goto statements
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    __int64 us_elapsed = 0;

    // Choose which GPU to run on, change this on a multi-GPU system. Then allocate GPU memory.
    {
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }

        cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }
    }
    
    t1 = Clock::now();
    for (int i = 0; i < loopCount; ++i) {
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        // Launch a kernel on the GPU with one thread for each element.
        addKernel << <blocks, threads >> > (dev_c, dev_a, dev_b, size);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        // NOTE: Below in the graph implementation this sync is included via graph dependencies 
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }
    t2 = Clock::now();
    us_elapsed = (__int64)(t2 - t1).count() / 1000;
    printf("Looped %d time(s) in %lld microseconds\n", loopCount, us_elapsed);
    *tElapsed = us_elapsed;

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cudaError_t addWithCudaGraph(int* c, const int* a, const int* b, unsigned int size, int loopCount, __int64* tElapsedGraph)
{
    // Original
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // For Graph
    cudaStream_t streamForGraph;
    cudaGraph_t graph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t memcpyNode, kernelNode;
    cudaKernelNodeParams kernelNodeParams = { 0 };
    cudaMemcpy3DParms memcpyParams = { 0 };

    // Choose which GPU to run on, change this on a multi-GPU system. Then allocate GPU memory.
    {
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }

        cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }
    }

    // Start of Graph Creation

    checkCudaErrors(cudaGraphCreate(&graph, 0));
    checkCudaErrors(cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking));

    // Add memcpy nodes for copying input vectors from host memory to GPU buffers

    memset(&memcpyParams, 0, sizeof(memcpyParams));

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr((void*)a, size * sizeof(int), size, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(dev_a, size * sizeof(float), size, 1);
    memcpyParams.extent = make_cudaExtent(size * sizeof(float), 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;

    checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
    nodeDependencies.push_back(memcpyNode);

    memset(&memcpyParams, 0, sizeof(memcpyParams));

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr((void*)b, size * sizeof(int), size, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(dev_b, size * sizeof(float), size, 1);
    memcpyParams.extent = make_cudaExtent(size * sizeof(float), 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;
        
    checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
    nodeDependencies.push_back(memcpyNode);

    // Add a kernel node for launching a kernel on the GPU

    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));

    kernelNodeParams.func = (void*)addKernel;
    kernelNodeParams.gridDim = dim3(blocks, 1, 1);
    kernelNodeParams.blockDim = dim3(threads, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    void* kernelArgs[4] = { (void*)&dev_c, (void*)&dev_a, (void*)&dev_b, &size };
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = NULL;

    checkCudaErrors(cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    // Add memcpy node for copying output vector from GPU buffers to host memory

    memset(&memcpyParams, 0, sizeof(memcpyParams));

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(dev_c, size * sizeof(int), size, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(c, size * sizeof(int), size, 1);
    memcpyParams.extent = make_cudaExtent(size * sizeof(int), 1, 1);
    memcpyParams.kind = cudaMemcpyDeviceToHost;
    checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams)); 

    if (VERBOSE) {
        cudaGraphNode_t* nodes = NULL;
        size_t numNodes = 0;
        checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
        printf("Num of nodes in the graph created manually = %zu\n", numNodes);
    }

    // Create an executable graph from a graph

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Run the graph

    auto t1 = Clock::now();
    for (int i = 0; i < loopCount; ++i) {
        checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
        checkCudaErrors(cudaStreamSynchronize(streamForGraph));
    }
    auto t2 = Clock::now();
    __int64 us_elapsed = (__int64)(t2 - t1).count() / 1000;
    printf("Looped %d time(s) in %lld microseconds\n", loopCount, us_elapsed);
    *tElapsedGraph = us_elapsed;

    // Clean up

    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaStreamDestroy(streamForGraph));

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
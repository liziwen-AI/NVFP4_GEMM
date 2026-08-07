#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Read-only kernel: coalesced float4 load with warp shuffle reduction
__global__ void read_only(const float4* __restrict__ src, size_t n, float4* block_sums) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll 4
    for (size_t i = idx; i < n; i += stride) {
        float4 v = src[i];
        sum.x += v.x;
        sum.y += v.y;
        sum.z += v.z;
        sum.w += v.w;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum.x += __shfl_down_sync(0xFFFFFFFF, sum.x, offset);
        sum.y += __shfl_down_sync(0xFFFFFFFF, sum.y, offset);
        sum.z += __shfl_down_sync(0xFFFFFFFF, sum.z, offset);
        sum.w += __shfl_down_sync(0xFFFFFFFF, sum.w, offset);
    }

    if ((threadIdx.x & 31) == 0) {
        block_sums[blockIdx.x] = sum;
    }
}

// Write-only kernel: coalesced float4 store
__global__ void write_only(float4* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float4 val = make_float4(1.0f, 2.0f, 3.0f, 4.0f);

    #pragma unroll 4
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = val;
    }
}

// Copy kernel: read + write
__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    #pragma unroll 4
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char** argv) {
    int device = 0;
    size_t size_mb = 8192;
    int iterations = 20;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            size_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-d device] [-s size_mb] [-n iterations]\n", argv[0]);
            printf("  -d: GPU device ID (default: 0)\n");
            printf("  -s: Buffer size in MB (default: 8192)\n");
            printf("  -n: Number of iterations (default: 20)\n");
            return 0;
        }
    }

    check_cuda(cudaSetDevice(device), "cudaSetDevice");

    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    printf("Device: %s\n", prop.name);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Buffer Size: %zu MB (%zu float4 elements)\n", size_mb, (size_mb * 1024ULL * 1024ULL) / sizeof(float4));
    printf("Iterations: %d\n\n", iterations);

    size_t n_float4 = (size_mb * 1024ULL * 1024ULL) / sizeof(float4);
    size_t bytes = n_float4 * sizeof(float4);

    float4 *d_src, *d_dst, *d_dummy;
    check_cuda(cudaMalloc(&d_src, bytes), "cudaMalloc d_src");
    check_cuda(cudaMalloc(&d_dst, bytes), "cudaMalloc d_dst");

    int blocks = prop.multiProcessorCount * 8;
    const int threads = 256;
    check_cuda(cudaMalloc(&d_dummy, blocks * sizeof(float4)), "cudaMalloc d_dummy");

    check_cuda(cudaMemset(d_src, 0, bytes), "cudaMemset d_src");
    check_cuda(cudaMemset(d_dst, 0, bytes), "cudaMemset d_dst");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Warmup
    for (int i = 0; i < 5; i++) {
        copy_kernel<<<blocks, threads>>>(d_src, d_dst, n_float4);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    printf("%-22s : Bandwidth\n", "Test");
    printf("---------------------------------------------------\n");

    // --- Read-only test ---
    for (int i = 0; i < 5; i++) {
        read_only<<<blocks, threads>>>(d_src, n_float4, d_dummy);
    }
    check_cuda(cudaDeviceSynchronize(), "read warmup");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; i++) {
        read_only<<<blocks, threads>>>(d_src, n_float4, d_dummy);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    {
        float ms = 0;
        check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        double bw = (double)bytes * iterations / (ms / 1000.0) / 1e12;
        printf("%-22s : %7.2f TB/s  (%8.1f GB/s)  [%.3f ms/iter]\n",
               "Read-only", bw, bw * 1000.0, ms / iterations);
    }

    // --- Write-only test ---
    for (int i = 0; i < 5; i++) {
        write_only<<<blocks, threads>>>(d_dst, n_float4);
    }
    check_cuda(cudaDeviceSynchronize(), "write warmup");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; i++) {
        write_only<<<blocks, threads>>>(d_dst, n_float4);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    {
        float ms = 0;
        check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        double bw = (double)bytes * iterations / (ms / 1000.0) / 1e12;
        printf("%-22s : %7.2f TB/s  (%8.1f GB/s)  [%.3f ms/iter]\n",
               "Write-only", bw, bw * 1000.0, ms / iterations);
    }

    // --- Copy test (read + write total) ---
    for (int i = 0; i < 5; i++) {
        copy_kernel<<<blocks, threads>>>(d_src, d_dst, n_float4);
    }
    check_cuda(cudaDeviceSynchronize(), "copy warmup");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; i++) {
        copy_kernel<<<blocks, threads>>>(d_src, d_dst, n_float4);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    {
        float ms = 0;
        check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        double bw_total = (double)bytes * 2 * iterations / (ms / 1000.0) / 1e12;
        double bw_read  = (double)bytes * iterations / (ms / 1000.0) / 1e12;
        printf("%-22s : %7.2f TB/s  (%8.1f GB/s)  [%.3f ms/iter]\n",
               "Copy (R+W total)", bw_total, bw_total * 1000.0, ms / iterations);
        printf("%-22s : %7.2f TB/s  (%8.1f GB/s)  [%.3f ms/iter]\n",
               "Copy (read comp.)", bw_read, bw_read * 1000.0, ms / iterations);
    }

    printf("\nB200 HBM3e theoretical peak: ~8.0 TB/s (read or write)\n");

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_dummy);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
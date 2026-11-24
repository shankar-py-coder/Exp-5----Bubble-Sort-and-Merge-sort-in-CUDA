# Exp5 Bubble Sort and Merge sort in CUDA
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```

%%writefile sorting.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

// Kernel for Bubble Sort
__global__ void bubbleSortKernel(int *d_arr, int n, int phase) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i = 2 * idx + phase;

    if (i + 1 < n) {
        if (d_arr[i] > d_arr[i + 1]) {
            int temp = d_arr[i];
            d_arr[i] = d_arr[i + 1];
            d_arr[i + 1] = temp;
        }
    }
}

// Kernel for Merge Sort
__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int left = idx * 2 * size;

    if (left >= n) return;

    int mid = (left + size < n) ? left + size : n;
    int right = (left + 2 * size < n) ? left + 2 * size : n;

    int i = left, j = mid, k = left;

    while (i < mid && j < right) {
        if (d_arr[i] <= d_arr[j])
            d_temp[k++] = d_arr[i++];
        else
            d_temp[k++] = d_arr[j++];
    }

    while (i < mid)
        d_temp[k++] = d_arr[i++];

    while (j < right)
        d_temp[k++] = d_arr[j++];
}

// Host function for merging arrays
void mergeHost(int *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];

    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];

    free(L);
    free(R);
}

// Bubble Sort on GPU
void bubbleSort(int *arr, int n) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads = 256;
    int blocks = (((n + 1) / 2) + threads - 1) / threads;

    for (int i = 0; i < n; i++) {
        bubbleSortKernel<<<blocks, threads>>>(d_arr, n, i % 2);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);
}

// Merge Sort on GPU
void mergeSort(int *arr, int n) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads = 256;

    for (int size = 1; size < n; size *= 2) {
        int count = (n + 2 * size - 1) / (2 * size);
        int blocks = (count + threads - 1) / threads;

        mergeSortKernel<<<blocks, threads>>>(d_arr, d_temp, n, size);

        int *temp = d_arr;
        d_arr = d_temp;
        d_temp = temp;
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_temp);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);
}

// Bubble Sort on CPU
void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Bubble Sort (CPU) took %f milliseconds\n", duration.count());
}

// Merge Sort on CPU
void mergeSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    int *temp = (int*)malloc(n * sizeof(int));

    for (int size = 1; size < n; size *= 2) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = (left + size < n) ? left + size : n;
            int right = (left + 2 * size < n) ? left + 2 * size : n;

            if (mid < right) {
                int i = left, j = mid, k = left;

                while (i < mid && j < right) {
                    if (arr[i] <= arr[j])
                        temp[k++] = arr[i++];
                    else
                        temp[k++] = arr[j++];
                }

                while (i < mid)
                    temp[k++] = arr[i++];

                while (j < right)
                    temp[k++] = arr[j++];

                for (int x = left; x < right; x++)
                    arr[x] = temp[x];
            }
        }
    }

    free(temp);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Merge Sort (CPU) took %f milliseconds\n", duration.count());
}

// Print array
void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    printf("\n");
}

// Main function
int main() {
    int n = 10000;
    int *arr = (int*)malloc(n * sizeof(int));

    // Generating random array
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    printf("Original array: \n");
    printArray(arr, n);

    // Bubble Sort CPU
    bubbleSortCPU(arr, n);

    printf("Sorted array using Bubble Sort (CPU): \n");
    printArray(arr, n);

    // Generating random array again for GPU
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    // Bubble Sort GPU
    bubbleSort(arr, n);

    printf("Sorted array using Bubble Sort (GPU): \n");
    printArray(arr, n);

    // Generating random array again for Merge Sort
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    printf("Original array: \n");
    printArray(arr, n);

    // Merge Sort CPU
    mergeSortCPU(arr, n);

    printf("Sorted array using Merge Sort (CPU): \n");
    printArray(arr, n);

    // Generating random array again for GPU
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    // Merge Sort GPU
    mergeSort(arr, n);

    printf("Sorted array using Merge Sort (GPU): \n");
    printArray(arr, n);

    free(arr);
    return 0;
}

```

## OUTPUT:
<img width="1732" height="425" alt="image" src="https://github.com/user-attachments/assets/32d69135-6a64-4ace-8392-cb6232ee4a51" />


## RESULT:
Thus, the program has been executed using CUDA to implement bubble and merge sort, and it shows the time of bubble sort CPU is took 266.073615 milliseconds and by GPU it took 27.924929 milliseconds and for merge sort it shows the time it took by CPU is 1.569596 milliseconds and for GPU it took 4.038688 milliseconds.

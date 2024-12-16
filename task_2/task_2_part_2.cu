#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <cub/cub.cuh>

void checkCuda(cudaError_t result, const char *func){
    if (result != cudaSuccess){
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

#define CUDA_CHECK(ans) { checkCuda((ans), #ans); }


__device__ bool condition_failed_check(int left_item, int right_item, bool is_increasing){
    bool condition_failed = false;
    if(left_item == right_item){
        condition_failed = true;
    }
    else if(is_increasing && (left_item > right_item)){
        condition_failed = true;
    }
    else if (!is_increasing && (left_item < right_item)){
        condition_failed = true;
    }
    else if((std::abs(left_item - right_item) > 3) || 
            (std::abs(left_item - right_item) < 1)){
        condition_failed = true;
    }

    return condition_failed;
}


__device__ bool isSafe(int* array, int n){
    bool condition_failed;
    bool is_increasing = array[0] < array[1];

    for(int i = 1; i <  n; i++){
        condition_failed = condition_failed_check(array[i-1], array[i], is_increasing);

        if (condition_failed){
            return false;
        }
    }
    return true;
}

__global__ void checkIfArrayIsSafe(int** arrays, int* is_safe_array, int n, int* len_of_arrays){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    bool is_increasing = arrays[idx][0] < arrays[idx][1];

    if (idx >= n){
        return;
    }

    bool condition_failed = false;


    bool is_safe = isSafe(arrays[idx], len_of_arrays[idx]);


    is_safe_array[idx] = int(is_safe);

    if(is_safe){
        return;
    }


    // // Now, call this function again, but check whether you can remove one level
    int temporary_memory[100];
    
    for(int i = 0; i < len_of_arrays[idx]; i++){
        for (int j = 0, k = 0; j < len_of_arrays[idx]; j++){
            if(j != i){
                temporary_memory[k++] = arrays[idx][j];
            }
        }
        is_safe = isSafe(temporary_memory, len_of_arrays[idx] - 1);

        if(is_safe){
            is_safe_array[idx] = 1;
            return;
        }
    }


    is_safe_array[idx] = 0;
}


// move get safe array code to a function
int* get_is_safe_array(int** arrays, int* len_of_arrays, int n){
    // move array to device
    int** d_arrays;
    int* d_len_of_arrays;

    // allocate memory for the array of pointers, which have dynamically allocated memory
    CUDA_CHECK(cudaMalloc(&d_arrays, n * sizeof(int*)));
    // CUDA_CHECK(cudaMemcpy(d_arrays, arrays, n * sizeof(int*), cudaMemcpyHostToDevice));
    for (int i = 0; i < n; i++){
        int* d_array;
        CUDA_CHECK(cudaMalloc(&d_array, len_of_arrays[i] * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_array, arrays[i], len_of_arrays[i] * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_arrays[i], &d_array, sizeof(int*), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_len_of_arrays, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_len_of_arrays, len_of_arrays, n * sizeof(int), cudaMemcpyHostToDevice));

    // allocate memory for the result
    int* is_safe_array = new int[n];
    int* d_is_safe_array;
    CUDA_CHECK(cudaMalloc(&d_is_safe_array, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_is_safe_array, 0, n * sizeof(int)));

    // calculate the number of blocks and threads
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // run the kernel
    checkIfArrayIsSafe<<<numBlocks, blockSize>>>(d_arrays, d_is_safe_array, n, d_len_of_arrays);

    // run debug
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy the result back
    CUDA_CHECK(cudaMemcpy(is_safe_array, d_is_safe_array, n * sizeof(int), cudaMemcpyDeviceToHost));

    return is_safe_array;
}


__global__ void sumArray(int* d_array, int n, int *result){
    extern __shared__ int shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tix = threadIdx.x;

    // if (idx < n) {
    //     printf("1st try: Block %d, Thread %d: d_array[%d] = %d\n", blockIdx.x, tix, idx, d_array[idx]);
    // }
    if(tix < blockDim.x && idx >= n){
        return;
    }
    // Load elements into shared memory
    if (idx < n){
        shared_data[tix] = d_array[idx];
    }
    else{
        shared_data[tix] = 0;
    }

    // Synchronize threads to make sure all elements are loaded
    __syncthreads(); 

    // if (idx < n) {
    //     printf("2nd try: Block %d, Thread %d: d_array[%d] = %d\n", blockIdx.x, tix, idx, d_array[idx]);
    // }
    // Perform the reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tix < stride) {
            shared_data[tix] += shared_data[tix + stride];
        }
        __syncthreads();
    }

    if(tix == 0){
        atomicAdd(result, shared_data[0]);
    }
}

int main(int argc, char** argv){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    // open the file
    std::cout << "Reading file: " << argv[1] << std::endl;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[1] << " <input_file_path>" << std::endl;
        return 1;
    }
    std::cout << "Reading file: " << argv[1] << std::endl;
    std::string filePath = argv[1];
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filePath << std::endl;
        return 1;
    }
    
    // read the number of elements

    // Use istreambuf_iterator to count newlines
    int n = std::count(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>(),
        '\n'
    );

    std::cout << "Number of lines: " << n << std::endl;

    // return file pointer to the beginning
    file.close();
    file.open(filePath);

    // make an array of arrays
    int** arrays = new int*[n];
    int* len_of_arrays = new int[n];

    for (int i = 0; i < n; i++){
        std::string read_line;
        std::getline(file, read_line);
        std::string delimiter = " ";

        // calculate number of elements in the line
        int num_elements = std::count(
            read_line.begin(),
            read_line.end(),
            ' '
        ) + 1;

        // allocate memory for the array
        arrays[i] = new int[num_elements];

        // read the elements of the array
        int pos;
        for (int j = 0; j < num_elements; j++){
            pos = read_line.find(delimiter);
            arrays[i][j] = std::stoi(read_line.substr(0, pos));
            read_line.erase(0, pos + delimiter.length());
        }

        len_of_arrays[i] = num_elements;
    }

    // close the file
    file.close();

    int* is_safe_array = get_is_safe_array(arrays, len_of_arrays, n);

    // print out the result
    for (int i = 0; i < n; i++){
        std::cout << "Result: " << is_safe_array[i];
        std::cout << " List:  ";

        // print full array
        for (int j = 0; j < len_of_arrays[i]; j++){
            std::cout << arrays[i][j] << " ";
        }
        std::cout << std::endl;
    }


    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    int how_many_safe = 0;
    int sharedMemSize = blockSize * sizeof(int);
    int* d_how_many_safe;
    int* d_is_safe_array;

    CUDA_CHECK(cudaMalloc(&d_is_safe_array, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_is_safe_array, is_safe_array, n * sizeof(int), cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMalloc(&d_how_many_safe, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_how_many_safe, 0, sizeof(int)));


    sumArray<<<numBlocks, blockSize, sharedMemSize>>>(d_is_safe_array, n, d_how_many_safe);

    // Move back to device
    CUDA_CHECK(cudaMemcpy(&how_many_safe, d_how_many_safe, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Number of safe arrays: " << how_many_safe << std::endl;


}
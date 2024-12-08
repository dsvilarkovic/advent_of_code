#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <cub/cub.cuh>
// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html#_CPPv4N3cub15DeviceRadixSortE

void checkCuda(cudaError_t result, const char *func){
    if (result != cudaSuccess){
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

int* returnListIndices(int n){
    int* list_indices = new int[n];
    for (int i = 0; i < n; i++){
        list_indices[i] = i;
    }
    return list_indices;
}


#define CUDA_CHECK(ans) { checkCuda((ans), #ans); }


int* sortArrayUsingCub(int* array, int n){
    // Allocate device memory
    int* d_array_key_in;
    int* d_array_key_out;
    int* list_indices = returnListIndices(n);

    CUDA_CHECK(cudaMalloc(&d_array_key_in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_array_key_out, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_array_key_in, array, n * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, 
        d_array_key_in, 
        d_array_key_out,     
        n);   
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Sort keys
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, 
        d_array_key_in, 
        d_array_key_out,
        n);

    // Copy keys back to host
    int* sorted_array = new int[n];
    CUDA_CHECK(cudaMemcpy(sorted_array, d_array_key_out, n * sizeof(int), cudaMemcpyDeviceToHost));


    // Free memory
    CUDA_CHECK(cudaFree(d_array_key_in));
    CUDA_CHECK(cudaFree(d_array_key_out));
    CUDA_CHECK(cudaFree(d_temp_storage));


    return sorted_array;
}

// Now CUDA summing array 
__global__ void arrayDistanceKernel(const int *a, const int *b, int *result, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n){
        // result[index] = a[index] + b[index];
        result[index] = std::abs(a[index] - b[index]);
    }
}

int* sumArrayDistanceUsingCuda(int* left_array, int* right_array, int n){
    // Allocate device memory
    // Apparently d_* prefix is for device/GPU variables
    int* d_left_array;
    int* d_right_array;
    int* d_result_array;
    int* result = new int[n];

    CUDA_CHECK(cudaMalloc(&d_left_array, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right_array, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result_array, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_left_array, left_array, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right_array, right_array, n * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel for summing 
    int threadsPerBlock = 256; // some random number which I have yet to learn it's upper/lower values to use
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    arrayDistanceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_left_array, d_right_array, d_result_array, n);

    // Copy the gpu result to the host
    // CUDA_CHECK(cudaMemcpy(result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result, d_result_array, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Free memory from the GPU
    CUDA_CHECK(cudaFree(d_left_array));
    CUDA_CHECK(cudaFree(d_right_array));
    CUDA_CHECK(cudaFree(d_result_array));

    return result;
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

    // allocate memory for the array
    int* left_array = new int[n];
    int* right_array = new int[n];
    int* result_array = new int[n];
    int pos;
    std::string delimiter = " ";

    // read the elements of the array
    std::string read_line;
    // int final_result = 0;
    for (int i = 0; i < n; i++){
        std::getline(file, read_line);

        pos = read_line.find(delimiter);
        left_array[i] = std::stoi(read_line.substr(0, pos));
        right_array[i] = std::stoi(read_line.substr(pos + 1, read_line.length() - pos - 1));
    }


    // close the file
    file.close();


    // Sorting the arrays
    int* sorted_left_array = sortArrayUsingCub(left_array, n);
    int* sorted_right_array = sortArrayUsingCub(right_array, n);


    // innit final_result but with 64 bit
    int final_result = 0;
    // CPU way of summing the array
    for (int i = 0; i < n; i++){
        result_array[i] = std::abs(sorted_left_array[i] - sorted_right_array[i]);
        final_result += result_array[i];
    }
    // GPU way of summing the array is too complicated for me at this point



    // print the result
    std::cout << "Final result: " << final_result << std::endl;

    return 0;
}
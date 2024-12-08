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

#define CUDA_CHECK(ans) { checkCuda((ans), #ans); }

// Now CUDA finding the intersection of two arrays
// we do this by counting the number of appearances of each element in the left_array in the right_array.
// and we don't save on memory use since we got a lot of GPU
__global__ void arrayAppearanceKernel(const int *key_array, const int *value_array, int n, int* similarity_score){
    // shared memory for the counter between threads
    extern __shared__ int shared_counts[]; 
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize shared memory
    if(tid < blockDim.x){
        shared_counts[tid] = 0;
    }
    __syncthreads();

    // counting number of appearances of element in array within block

    if (i < n){
        int counter = 0; 
        for (int j = 0; j < n; j++){
            if (key_array[i] == value_array[j]){
                counter++;
            }
        }
        shared_counts[tid] = counter;
    }
    __syncthreads();

    // if(i < n){
        // atomicAdd(similarity_score, shared_counts[tid] * key_array[blockIdx.x * blockDim.x + tid]);
    // }
    // calculate similarity score, which is num_appearances * element
    // add to similarity score using atomicAdd within a block
    if (tid == 0){
        int block_sum = 0;

        for (int k = 0; k < blockDim.x; k++){
            if(blockIdx.x * blockDim.x + k < n){
                block_sum += shared_counts[k] * key_array[blockIdx.x * blockDim.x + k];
            }
        }
        atomicAdd(similarity_score, block_sum);
    }


}

int obtainSimilarityScore(int* left_array, int* right_array, int n){
    int* d_left_array;
    int* d_right_array;
    int similarity_score = 0;
    int* d_similarity_score;

    // allocate memory for the arrays
    CUDA_CHECK(cudaMalloc(&d_left_array, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right_array, n * sizeof(int)));
    // allocate memory for the similarity score, and initialize it with 0
    CUDA_CHECK(cudaMalloc(&d_similarity_score, sizeof(int)));

    // initialize d_counter_array with all zeros
    CUDA_CHECK(cudaMemcpy(d_left_array, left_array, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right_array, right_array, n * sizeof(int), cudaMemcpyHostToDevice));

    // Launch the kernel
    arrayAppearanceKernel<<<(n + 255) / 256, 256, n * sizeof(int)>>>(d_left_array, d_right_array, n, d_similarity_score);

    cudaDeviceSynchronize();
    // Copy the result back to the host    

    CUDA_CHECK(cudaMemcpy(&similarity_score, d_similarity_score, sizeof(int), cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_CHECK(cudaFree(d_left_array));
    CUDA_CHECK(cudaFree(d_right_array));
    CUDA_CHECK(cudaFree(d_similarity_score));

    return similarity_score;
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
    // file.open("input_level_1.txt");

    // allocate memory for the array
    int* left_array = new int[n];
    int* right_array = new int[n];
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

    int similarity_score;

    similarity_score = obtainSimilarityScore(left_array, right_array, n);

    std::cout << "Similarity score: " << similarity_score << std::endl;

    return 0;
}
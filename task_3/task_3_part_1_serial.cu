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

__device__ char* my_strstr(char* haystack, const char* needle) {
    if (!*needle) return haystack;
    for (; *haystack; ++haystack) {
        if (*haystack == *needle) {
            const char* h = haystack, *n = needle;
            for (; *h && *n && *h == *n; ++h, ++n);
            if (!*n) return haystack;
        }
    }
    return nullptr;
}

__device__ char* my_strchr(char* str, int c) {
    while (*str) {
        if (*str == c) return str;
        ++str;
    }
    return nullptr;
}

__device__ long my_strtol(const char* str, char** endptr, int base) {
    long result = 0;
    while (*str >= '0' && *str <= '9') {
        result = result * base + (*str - '0');
        ++str;
    }
    if (endptr) *endptr = (char*)str;
    return result;
}

__device__ void my_strncpy(char* dest, const char* src, size_t n) {
    for (size_t i = 0; i < n && src[i] != '\0'; ++i) {
        dest[i] = src[i];
    }
    dest[n] = '\0';
}

// "mul" + "()" + 3 digits + "," + 3 digits
// 3 + 2 + 3 + 1 + 3 = 12 is maximum length to consider

// buffer is the full file
// start_of_buffer is the starting position of the buffer
// considered_length is the length of the buffer to consider
__global__ void checkString(char* buffer, int considered_length, int buffer_length, int* total_result){
    // get the thread index
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int start_of_buffer = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("start_of_buffer: %d\n", start_of_buffer);
    // print start_of_buffer, considered_length, buffer_length
    // printf("start_of_buffer: %d, considered_length: %d, buffer_length: %d\n", start_of_buffer, considered_length, buffer_length);
    if(start_of_buffer + considered_length > buffer_length){
        // printf("Thread %d didn't work \n", start_of_buffer);
        return;
    }

    // get mul
    // printf("start_of_buffer: %d\n", start_of_buffer);
    // std::string buffer_str(buffer);
    char* buffer_start = buffer + start_of_buffer;
    // print first 5 characters of buffer

    // printf("Thread %d, first 5 characters: %.5s\n", start_of_buffer, buffer_start);

    char* mul_pos = my_strstr(buffer_start, "mul");

    // print first 5 characters
    
    
    if (mul_pos == nullptr || mul_pos - buffer_start > considered_length){
        return;
    }
    // get the next character
    // int next_char_pos = mul_pos + 3;
    char* next_char_pos = mul_pos + 3;

    if(*next_char_pos != '('){
        return;
    }

    // get the comma
    char* comma_pos = my_strstr(next_char_pos, ",");

    // printf("comma_pos: %p\n", comma_pos);
    if (mul_pos - buffer_start != 0 || comma_pos == nullptr || comma_pos - buffer_start > considered_length){
        return;
    }

    // get the first number
    // int opening_bracket_pos = buffer_str.find("(", next_char_pos);
    char* opening_bracket_pos = my_strstr(next_char_pos, "(");
    char first_num_str[10];
    my_strncpy(first_num_str, opening_bracket_pos + 1, comma_pos - opening_bracket_pos - 1);
    first_num_str[comma_pos - opening_bracket_pos - 1] = '\0';
    char* end_ptr;
    long first_num_int = my_strtol(first_num_str, &end_ptr, 10);
    if (*end_ptr != '\0') {
        return;
    }

    // get the second number
    char* closing_bracket_pos = my_strchr(comma_pos, ')');

    if (closing_bracket_pos == nullptr || closing_bracket_pos - buffer_start > considered_length){
        return;
    }

    char second_num_str[10];
    my_strncpy(second_num_str, comma_pos + 1, closing_bracket_pos - comma_pos - 1);
    second_num_str[closing_bracket_pos - comma_pos - 1] = '\0';

    long second_num_int = my_strtol(second_num_str, &end_ptr, 10);
    if (*end_ptr != '\0') {
        return;
    }

    if(first_num_int * second_num_int > 0){
        printf("Thread %ld, command: {%.12s} mul_pos {%d} found equation mul(%ld, %ld)=%ld \n", start_of_buffer, buffer_start, mul_pos - buffer_start, first_num_int, second_num_int, first_num_int * second_num_int);
    }

    atomicAdd(total_result, first_num_int * second_num_int);
}

int main(int argc, char** argv){

    // // read full file input_level_3.txt into buffer
    std::ifstream file(argv[1]);
    std::string buffer;

    if (file.is_open()){
        file.seekg(0, std::ios::end); // seek to end of file
        buffer.resize(file.tellg()); // get file size
        file.seekg(0); // seek to beginning of file
        file.read(&buffer[0], buffer.size()); // read file into buffer
        file.close();
    }


    // // print buffer
    // std::cout << buffer << std::endl;

    // move through the buffer for each string "mul"
    int buffer_pos = 0;
    // get length of buffer
    int buffer_length = buffer.length();
    int opening_bracket_pos = 0;
    int comma_pos = 0;
    int closing_bracket_pos = 0;
    long total_result = 0;
    
    while(true){
        // find the next "mul" in the buffer
        buffer_pos = buffer.find("mul", buffer_pos);
        if (buffer_pos == -1){
            break;
        }
        // after mul is found, check if next character is "("
        opening_bracket_pos = buffer.find("(", buffer_pos);

        if(buffer_pos + 3 < buffer_length && buffer[buffer_pos + 3] == '('){

            comma_pos = buffer.find(",", buffer_pos);
            if (comma_pos != -1){
                // get the first number 
                std::string first_num = buffer.substr(opening_bracket_pos + 1, comma_pos - opening_bracket_pos - 1);
                bool is_number = std::all_of(first_num.begin(), first_num.end(), ::isdigit);

                if (is_number){
                    std::cout << "first_num: " << first_num << std::endl;
                
                    // try to convert to int
                    int first_num_int = std::stoi(first_num);
                    // std::cout << "first_num_int: " << first_num_int << std::endl;
                    // get the second number


                    closing_bracket_pos = buffer.find(")", comma_pos);

                    if(closing_bracket_pos == -1){
                        break;
                    }
                    std::string second_num = buffer.substr(comma_pos + 1, closing_bracket_pos - comma_pos - 1);

                    is_number = std::all_of(second_num.begin(), second_num.end(), ::isdigit);

                    if(is_number){
                        std::cout << "second_num: " << second_num << std::endl;
                        int second_num_int = std::stoi(second_num);
                        std::cout << "second_num_int: " << second_num_int << std::endl;

                        // multiply the two numbers
                        int result = first_num_int * second_num_int;
                        std::cout << "result: " << result << std::endl;

                        total_result += result;
                    }

                }


            }
        }
        buffer_pos += 3; // move past "mul"
    }
    
    std::cout << "total_result: " << total_result << std::endl;
}
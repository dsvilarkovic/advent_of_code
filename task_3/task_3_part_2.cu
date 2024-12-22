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

int main(int argc, char** argv){

    // read full file input_level_3.txt into buffer
    std::ifstream file(argv[1]);
    std::string buffer;

    if (file.is_open()){
        file.seekg(0, std::ios::end); // seek to end of file
        buffer.resize(file.tellg()); // get file size
        file.seekg(0); // seek to beginning of file
        file.read(&buffer[0], buffer.size()); // read file into buffer
        file.close();
    }

    // move through the buffer for each string "mul"
    int buffer_pos = 0;
    int buffer_length = buffer.length();
    int opening_bracket_pos = 0;
    int comma_pos = 0;
    int closing_bracket_pos = 0;
    long total_result = 0;
    bool mul_enabled = true; // mul instructions are enabled by default

    while(true){
        // find the next "mul", "do", or "don't" in the buffer
        int mul_pos = buffer.find("mul", buffer_pos);
        int do_pos = buffer.find("do()", buffer_pos);
        int dont_pos = buffer.find("don't()", buffer_pos);

        // find the earliest position among mul, do, and don't
        int next_pos = std::min({mul_pos, do_pos, dont_pos});

        if (next_pos == -1){
            break;
        }

        if (next_pos == mul_pos){
            // handle mul instruction
            buffer_pos = mul_pos;
            opening_bracket_pos = buffer.find("(", buffer_pos);

            if(buffer_pos + 3 < buffer_length && buffer[buffer_pos + 3] == '('){
                comma_pos = buffer.find(",", buffer_pos);
                if (comma_pos != -1){
                    std::string first_num = buffer.substr(opening_bracket_pos + 1, comma_pos - opening_bracket_pos - 1);
                    bool is_number = std::all_of(first_num.begin(), first_num.end(), ::isdigit);

                    if (is_number){
                        int first_num_int = std::stoi(first_num);
                        closing_bracket_pos = buffer.find(")", comma_pos);

                        if(closing_bracket_pos != -1){
                            std::string second_num = buffer.substr(comma_pos + 1, closing_bracket_pos - comma_pos - 1);
                            is_number = std::all_of(second_num.begin(), second_num.end(), ::isdigit);

                            if(is_number){
                                int second_num_int = std::stoi(second_num);
                                if (mul_enabled){
                                    int result = first_num_int * second_num_int;
                                    total_result += result;
                                }
                            }
                        }
                    }
                }
            }
            buffer_pos += 3; // move past "mul"
        } else if (next_pos == do_pos){
            // handle do() instruction
            mul_enabled = true;
            buffer_pos = do_pos + 4; // move past "do()"
        } else if (next_pos == dont_pos){
            // handle don't() instruction
            mul_enabled = false;
            buffer_pos = dont_pos + 7; // move past "don't()"
        }
    }

    std::cout << "total_result: " << total_result << std::endl;
}
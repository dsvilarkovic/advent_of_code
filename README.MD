# Advent of Code (CUDA edition)

My personal attempt in trying to learn CUDA through AoC tasks: https://adventofcode.com/

I probably won't do all the tasks, since I have other things to do in my life, but let's see how far I can get.

## Content
For compiling tasks, go for
```
bash compile_and_run_file.sh <task_to_run> <input_file_to_run>
```
For example for 1st part of Task 1:
```
bash compile_and_run_file.sh task_1/task_1_part_2.cu task_1/input_level_1.txt
```
### [Task 1](task_1)
Using basic CUDA commands for parallelization (threadidx, blockidx, synchronization, shared variables etc.). Sorting with CUB's RadixSort.

Code is split into two subtasks. I didn't attempt to make this clean and reusable, but the output results are correct according to website.
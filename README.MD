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
- Using basic CUDA commands for parallelization (threadidx, blockidx, synchronization, shared variables etc.). 
- Sorting with CUB's RadixSort.

Code is split into two subtasks. I didn't attempt to make this clean and reusable, but the output results are correct according to website.

### [Task 2](task_2)


#### First part: 
Task is about whether an 1d array inside a 2d array (N 1d arrays exist inside a 2d array, but each of it's own length) is safe by some particular criteria.
We can do this by scheduling in parallel each of the 1d arrays to be checked by some criteria and with the shared bool array of length N, returning a bool value for index at which array is fulfilling this criteria or not.
Bool array is then summed and reduced inside a GPU as well for number of 1D arrays fulfilling this criteria.

- Summing of array by using shared memory in blocks. 
- Atomic add. 
- Figured out there is a GPU printf for debugging. 
- Used for the first time 2d array (double pointer) for allocation of memory.

#### Second part:
As the first part, task is about whether an 1d array inside a 2d array (N 1d arrays exist inside a 2d array, but each of it's own length) is safe by some particular criteria. 
But this time with the twist! We need to check for each 1d array whether removing. I wanted to do something fancy. Instead I opted in for just looping through array and allocating n-1 array to check for. I wanted to do this in parallel, but CUDA mentions some dynamic allocation which I am not touching yet.
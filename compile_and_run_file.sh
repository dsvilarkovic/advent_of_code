# use single .cu file as input
input_file=$1
input_text_file=$2

# split the input file name by '.' and get the first part
input_file=$(echo $input_file | cut -f 1 -d '.')

# remove the previous output file
rm -f ${input_file}.out
rm -f ${input_file}.o

nvcc -c $1 -o ${input_file}.o 
nvcc ${input_file}.o -o ${input_file}.out

# Change to the directory containing the input file
./${input_file}.out $input_text_file

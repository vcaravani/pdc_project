using CuArrays
using CuArrays.CUSPARSE
using SparseArrays
# dimensions
N = 20
M = 10
p = 0.1

# create Martrices
A = sprand(N,M,p)
B = sprand(N,M,p)

# convert A,B to CSC format and move them to the GPU
d_A = CuSparseMatrixCSC(A);
d_B = CuSparseMatrixCSC(B);

# Perform d_A*d_B
d_C = CUSPARSE.gemm('N','T',d_A,d_B,'O','O','O')

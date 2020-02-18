using LinearAlgebraicRepresentation, SparseArrays, DataStructures
Lar = LinearAlgebraicRepresentation

function CV2FV( v::Array{Int64} )
	faces = [
		[v[1], v[2], v[3], v[4]], [v[5], v[6], v[7], v[8]],
		[v[1], v[2], v[5], v[6]],	[v[3], v[4], v[7], v[8]],
		[v[1], v[3], v[5], v[7]], [v[2], v[4], v[6], v[8]]]
end

function CV2EV( v::Array{Int64} )
	edges = [
		[v[1], v[2]], [v[3], v[4]], [v[5], v[6]], [v[7], v[8]], [v[1], v[3]], [v[2], v[4]],
		[v[5], v[7]], [v[6], v[8]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]], [v[4], v[8]]]
end

function K( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return sparse(I,J,Vals)
end

function K_GBLAS( CV )
	I = (vcat( [[k for h in CV[k]] for k=1:length(CV) ]...))
	J = vcat(CV...)
	X = [1 for k=1:length(I)]

	M = GrB_Matrix(OneBasedIndex.(I), OneBasedIndex.(J), X)
	return M
end

function collect_matrix_gblas(GBLAS_M)

	n_row = Int64(GrB_Matrix_nrows(GBLAS_M))
	n_col = Int64(GrB_Matrix_ncols(GBLAS_M))
	I1 = OneBasedIndex.([i for i=1:n_row])
	J1 = OneBasedIndex.([j for j=1:n_col])



	OUT = GrB_Matrix{Int64}()
	GrB_Matrix_new(OUT, GrB_INT64, n_row, n_col)
	GrB_Matrix_extract(OUT, GrB_NULL, GrB_NULL, GBLAS_M, I1, n_row, J1, n_col, GrB_NULL)

	I, J, X = GrB_Matrix_extractTuples(GBLAS_M)
	print((I))

	#=
	I =
	J =
	X =
	=#
	#return sparse(I,J,X)
end


using SuiteSparseGraphBLAS, GraphBLASInterface
GrB_init(GrB_NONBLOCKING)

using LinearAlgebra
#Matrix{Float64}(I, 2, 2)


V,CV = Lar.cuboidGrid([300,200,10])

VV = [[v] for v=1:size(V,2)]
FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))
EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))


# create M_0
M_0_test = K(VV)
M_0 = K_GBLAS(VV);
#@GxB_fprint(M_0, GxB_COMPLETE)

M_1_test = K(EV)
M_1 = K_GBLAS(EV);
#@GxB_fprint(M_1, GxB_COMPLETE)

n_row = Int64(GrB_Matrix_nrows(M_0))
n_col = Int64(GrB_Matrix_nrows(M_1))
I1 = [i for i=1:n_row-1]
J1 = [j for j=1:n_col-1]
x = [1 for i in 1:n_row]
sigma_1 = GrB_Matrix(ZeroBasedIndex.(I1),ZeroBasedIndex.(J1),x )



desc = GrB_Descriptor()
GrB_Descriptor_new(desc)
GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN)
GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE)

@btime sigma_1_test = M_0_test * M_1_test'
@btime GrB_mxm(sigma_1, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, M_0, M_1, desc)

'''
133.771 μs (21 allocations: 366.36 KiB)
49.671 μs (2 allocations: 32 bytes)

elapsed time (ns): 11039975
bytes allocated:   12461120
pool allocs:       17
malloc() calls:    7
realloc() calls:   2
  0.004908 seconds (6 allocations: 192 bytes)

elapsed time (ns): 4907548
bytes allocated:   192
pool allocs:       6





2.937 μs (15 allocations: 7.78 KiB)
3.381 μs (2 allocations: 32 bytes)
133.771 μs (21 allocations: 366.36 KiB)
49.671 μs (2 allocations: 32 bytes)
963.753 μs (21 allocations: 2.30 MiB)
483.015 μs (2 allocations: 32 bytes)
0.887698 seconds (26 allocations: 218.619 MiB, 8.43% gc time)
0.356622 seconds (6 allocations: 192 bytes)
774.199 ms (22 allocations: 218.62 MiB)
256.647 ms (2 allocations: 32 bytes)


'''

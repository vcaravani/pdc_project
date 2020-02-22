using LinearAlgebraicRepresentation, SparseArrays
Lar = LinearAlgebraicRepresentation

using SuiteSparseGraphBLAS, GraphBLASInterface, Test
GrB_init(GrB_NONBLOCKING)

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

function ZeroBasedIndex2Int(Indices::Array{ZeroBasedIndex,1})
	I = Int64[]

	for index in Indices
		index_int = replace(string(index), "ZeroBasedIndex(" => "")
		index_int = replace(index_int, ")" => "")
		index_int = replace(index_int, "x" => "")
		i = parse(Int, index_int, base=16)
		push!(I,i+1)
	end

	return I
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

	I = ZeroBasedIndex2Int(I)
	J = ZeroBasedIndex2Int(J)


	return sparse(I,J,X)
end



function intdivbyn_sigma(sigma,n)

	function integer_division(a)
    	return a ÷ n
	end

	INTDIV_BYN = GrB_UnaryOp()
	GrB_UnaryOp_new(INTDIV_BYN, integer_division, GrB_INT64, GrB_INT64)

	sigma_divbyn = GrB_Matrix{Int64}()
	GrB_Matrix_new(sigma_divbyn, GrB_INT64, GrB_Matrix_nrows(sigma), GrB_Matrix_ncols(sigma))

	GrB_Matrix_apply(sigma_divbyn, GrB_NULL, GrB_NULL, INTDIV_BYN, sigma, GrB_NULL)

	return sigma_divbyn

end

function floatdivbyn_sigma(sigma,n)

	function float_division(a)
    	return a / n
	end

	INTDIV_BYN = GrB_UnaryOp()
	GrB_UnaryOp_new(INTDIV_BYN, float_division, GrB_FP64, GrB_FP64)

	sigma_divbyn = GrB_Matrix{Float64}()
	GrB_Matrix_new(sigma_divbyn, GrB_FP64, GrB_Matrix_nrows(sigma), GrB_Matrix_ncols(sigma))

	GrB_Matrix_apply(sigma_divbyn, GrB_NULL, GrB_NULL, INTDIV_BYN, sigma, GrB_NULL)

	return sigma_divbyn

end







V,CV = Lar.cuboidGrid([3,2,1])

VV = [[v] for v=1:size(V,2)]
FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))
EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))


# create M_0
M_0_test = K(VV)
M_0 = K_GBLAS(VV);
# @info @GxB_fprint(M_0, GxB_COMPLETE)
#@test M_0_test == collect_matrix_gblas(M_0)


# create M_1
M_1_test = K(EV)
M_1 = K_GBLAS(EV);
#@GxB_fprint(M_1, GxB_COMPLETE)
#@test M_1_test == collect_matrix_gblas(M_1)

# init GrB_Descriptor for mxm
desc = GrB_Descriptor()
GrB_Descriptor_new(desc)
GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN)
GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE)

# init sigma_1
n_row = Int64(GrB_Matrix_nrows(M_0))
n_col = Int64(GrB_Matrix_nrows(M_1))
I1 = [i for i=1:n_row-1]
J1 = [j for j=1:n_col-1]
x = [1 for i in 1:n_row]
sigma_1 = GrB_Matrix(ZeroBasedIndex.(I1),ZeroBasedIndex.(J1),x )

# compute sigma_1
sigma_1_test = M_0_test * M_1_test'
@btime GrB_mxm(sigma_1, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, M_0, M_1, desc);
#@test sigma_1_test == collect_matrix_gblas(sigma_1)


# create M_2
M_2_test = K(FV)
M_2 = K_GBLAS(FV);
#@GxB_fprint(M_1, GxB_COMPLETE)
#@test M_2_test == collect_matrix_gblas(M_2)

# init sigma_2
n_row = Int64(GrB_Matrix_nrows(M_1));
n_col = Int64(GrB_Matrix_nrows(M_2));
I1 = [i for i=1:n_row-1]
J1 = [j for j=1:n_col-1]
x = [1 for i in 1:n_row]
sigma_2_notbinary = GrB_Matrix(ZeroBasedIndex.(I1),ZeroBasedIndex.(J1),x );


# compute sigma_2
sigma_2_test = M_1_test * M_2_test'
sigma_2_test = sigma_2_test .÷ 2

t2_1 = @btime GrB_mxm(sigma_2_notbinary, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, M_1, M_2, desc)
factor = 2;
#sigma_2 = normalize_sigma(sigma_2_notbinary, factor); # sigma_2 have zeros in x
t2_2 =  @btime sigma_2 = intdivbyn_sigma($sigma_2_notbinary,$factor)

t2 = t2_1+t2_2
@test sigma_2_test == collect_matrix_gblas(sigma_2)



# create M_3
M_3_test = K(CV);
M_3 = K_GBLAS(CV);
#@GxB_fprint(M_3, GxB_COMPLETE)
@test M_3_test == collect_matrix_gblas(M_3)

# init sigma_3
n_row = Int64(GrB_Matrix_nrows(M_2));
n_col = Int64(GrB_Matrix_nrows(M_3));
I1 = [i for i=1:n_row-1]
J1 = [j for j=1:n_col-1]
x = [1 for i in 1:n_row]
sigma_3_notbinary = GrB_Matrix(ZeroBasedIndex.(I1),ZeroBasedIndex.(J1),x );


# compute sigma_3
sigma_3_test = M_2_test * M_3_test'
sigma_3_test = sigma_3_test ./ 4
sigma_3_test = sigma_3_test .÷ 1

@Base.elapsed @btime GrB_mxm(sigma_3_notbinary, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, M_2, M_3, desc);
#factor = 0.25;
#sigma_3 = normalize_sigma_float(sigma_3_notbinary, factor);
#sigma_3 =  normalize_sigma(sigma_3, 1);
@Base.elapsed @btime sigma_3 = floatdivbyn_sigma(sigma_3_notbinary,4)
sigma_3 = intdivbyn_sigma(sigma_3, 1)
@test sigma_3_test == collect_matrix_gblas(sigma_3)


@GxB_fprint(sigma_3_notbinary, GxB_COMPLETE)

s1 = collect_matrix_gblas(sigma_1)
s2 = collect_matrix_gblas(sigma_2)
s3 = collect_matrix_gblas(sigma_3)

S2 = sum(s3,dims=2)


inner = [k for k=1:length(S2) if S2[k]==2]
outer = setdiff(collect(1:length(FV)), inner)

using ViewerGL
GL = ViewerGL

GL.VIEW([ GL.GLGrid(V,EV,GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[inner],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[outer],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);


# non possiamo dividere per 4 altrimenti non funziona, servono i due!!!

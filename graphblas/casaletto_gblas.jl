using LinearAlgebraicRepresentation, SparseArrays
Lar = LinearAlgebraicRepresentation

using ViewerGL
GL = ViewerGL

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
	I = (vcat( [[k for h in CV[k]] for k=1:length(CV) ]...));
	J = vcat(CV...);
	X = [1 for k=1:length(I)];

	M = GrB_Matrix(OneBasedIndex.(I), OneBasedIndex.(J), X);
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

	n_row = Int64(GrB_Matrix_nrows(GBLAS_M));
	n_col = Int64(GrB_Matrix_ncols(GBLAS_M));
	I1 = OneBasedIndex.([i for i=1:n_row]);
	J1 = OneBasedIndex.([j for j=1:n_col]);

	OUT = GrB_Matrix{Int64}();
	GrB_Matrix_new(OUT, GrB_INT64, n_row, n_col);
	GrB_Matrix_extract(OUT, GrB_NULL, GrB_NULL, GBLAS_M, I1, n_row, J1, n_col, GrB_NULL);

	I, J, X = GrB_Matrix_extractTuples(GBLAS_M);

	I = ZeroBasedIndex2Int(I);
	J = ZeroBasedIndex2Int(J);


	return sparse(I,J,X)
end

const integer_division_by_2 = a -> a ÷ 2
const integer_division_by_1 = a -> a ÷ 1
const float_division_by_4 = a ->  a / 4.0


function intdivbyn_sigma(sigma, f_div)

	INTDIV_BYN = GrB_UnaryOp()
	GrB_UnaryOp_new(INTDIV_BYN, f_div, GrB_INT64, GrB_INT64)

	sigma_divbyn = GrB_Matrix{Int64}()
	GrB_Matrix_new(sigma_divbyn, GrB_INT64, GrB_Matrix_nrows(sigma), GrB_Matrix_ncols(sigma))

	GrB_Matrix_apply(sigma_divbyn, GrB_NULL, GrB_NULL, INTDIV_BYN, sigma, GrB_NULL)

	return sigma_divbyn

end

function floatdivbyn_sigma(sigma,f_div)

	FLOATDIV_BYN = GrB_UnaryOp()
	GrB_UnaryOp_new(FLOATDIV_BYN, f_div, GrB_FP64, GrB_FP64)

	sigma_divbyn = GrB_Matrix{Float64}()
	GrB_Matrix_new(sigma_divbyn, GrB_FP64, GrB_Matrix_nrows(sigma), GrB_Matrix_ncols(sigma))

	GrB_Matrix_apply(sigma_divbyn, GrB_NULL, GrB_NULL, FLOATDIV_BYN, sigma, GrB_NULL)

	return sigma_divbyn

end



function init_descriptor_mxm()

	desc = GrB_Descriptor();
	GrB_Descriptor_new(desc);
	GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN);
	GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);

	return desc
end

function init_∂(A, B)

	n_row = Int64(GrB_Matrix_nrows(A));
	n_col = Int64(GrB_Matrix_nrows(B));
	I1 = [i for i=1:n_row-1];
	J1 = [j for j=1:n_col-1];
	x = [1 for i in 1:n_row];
	∂ = GrB_Matrix(ZeroBasedIndex.(I1),ZeroBasedIndex.(J1),x );

	return ∂

end


V,CV = Lar.cuboidGrid([3,2,1])
V,CV = random3cells([40,20,10],4_000)


VV = [[v] for v=1:size(V,2)];
FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))));
EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))));

#
M_0 = K_GBLAS(VV);
M_1 = K_GBLAS(EV);
M_2 = K_GBLAS(FV);
M_3 = K_GBLAS(CV);


# init GrB_Descriptor for mxm
desc = init_descriptor_mxm();

# compute ∂_1
∂_1 = init_∂(M_0,M_1);
GrB_mxm(∂_1, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, M_0, M_1, desc)
# compute ∂_2
∂_2 = init_∂(M_1,M_2);
GrB_mxm(∂_2, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, M_1, M_2, desc)
∂_2 = intdivbyn_sigma(∂_2,integer_division_by_2);

# compute ∂_3
∂_3 = init_∂(M_2,M_3);
GrB_mxm(∂_3, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, M_2, M_3, desc)
∂_3 = floatdivbyn_sigma(∂_3,float_division_by_4)
∂_3 = intdivbyn_sigma(∂_3, integer_division_by_1)


s1 = collect_matrix_gblas(∂_1)
s2 = collect_matrix_gblas(∂_2)
s3 = collect_matrix_gblas(∂_3)

S2 = sum(s3,dims=2)
inner = [k for k=1:length(S2) if S2[k]==2]
outer = setdiff(collect(1:length(FV)), inner)



GL.VIEW([ GL.GLGrid(V,EV,GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[inner],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[outer],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

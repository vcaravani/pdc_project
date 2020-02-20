using LinearAlgebraicRepresentation, SparseArrays
Lar = LinearAlgebraicRepresentation

using CuArrays,CuArrays.CUSPARSE

using SuiteSparseGraphBLAS, GraphBLASInterface, Test
GrB_init(GrB_NONBLOCKING)

using BenchmarkTools

using ViewerGL
GL = ViewerGL


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


function K_sparse( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = Float32[1 for k=1:length(I)]
	CV_sparse = sparse(I,J,Vals)
	d_CV = CuSparseMatrixCSR(CV_sparse)

	return d_CV
end

function K_f( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return sparse(I,J,Vals)
end

function K_cuda( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return cu(sparse(I,J,Vals))
end

n = 3;
m = 2;
k = 1;

V,CV = Lar.cuboidGrid([n,m,k]);

VV = [[v] for v=1:size(V,2)];
EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))));
FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))));


M_0 = K_f(VV);
M_1 = K_f(EV);
M_2 = K_f(FV);
M_3 = K_f(CV);



@btime sigma_1 =  M_0 * M_1';
@btime sigma_2 = M_1 * M_2' .÷ 2;
@btime sigma_3 = M_2 * M_3' .÷ 4;



M_0_cu = cu(M_0);
M_1_cu = cu(M_1);
M_2_cu = cu(M_2);
M_3_cu = cu(M_3);

#CuArrays.allowscalar(false)

@btime sigma_1_cu = M_0_cu * M_1_cu';
@btime sigma_2_cu = M_1_cu * M_2_cu' .÷ 2;
@btime sigma_3_cu = (M_2_cu * M_3_cu') .÷ 4;



M_0_cusparse = K_sparse(VV);
M_1_cusparse = K_sparse(EV);
M_2_cusparse = K_sparse(FV);
M_3_cusparse = K_sparse(CV);


@btime sigma_1_cusparse = CUSPARSE.gemm('N','T',M_0_cusparse,M_1_cusparse,'O','O','O');
@btime sigma_2_cusparse = CUSPARSE.gemm('N','T',M_1_cusparse,M_2_cusparse,'O','O','O');
@btime sigma_3_cusparse = CUSPARSE.gemm('N','T',M_2_cusparse,M_3_cusparse,'O','O','O');

@btime sigma_2_cusparse = cu(collect((CUSPARSE.gemm('N','T',M_1_cusparse,M_2_cusparse,'O','O','O')))) .÷ 2;
@btime sigma_3_cusparse = collect(CUSPARSE.gemm('N','T',M_2_cusparse,M_3_cusparse,'O','O','O')) .÷ 4;




cutimes, jltimes, cusparsetimes, graphtimes = Float64[], Float64[], Float64[], Float64[]

function benchmark(n,m,k)

	V,CV = Lar.cuboidGrid([n,m,k])
	VV = [[v] for v=1:size(V,2)]
	EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))
	FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))
	
	t1,t2,t3 = 0.0, 0.0, 0.0 #where 0.0 = not defined
	
    K_cpu, K_gpu, K_gpu_sparse, K_gblas = K_f, K_cuda, K_sparse, K_GBLAS

			for K in (K_cpu,K_gpu,K_gpu_sparse, K_gblas)

				M_0 = K(VV)
				M_1 = K(EV)
				M_2 = K(FV)
				M_3 = K(CV)
				
				print("\n")
				@info ("Benchmarking chain complex generation with: ") 
				print(string(typeof(M_0))*"\n")
				
				if K == K_gblas
					
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
					@info("M0*M1'")
					t1 = Base.@elapsed @btime GrB_mxm($sigma_1, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, $M_0, $M_1, $desc);
					#@GxB_fprint(sigma_1, GxB_COMPLETE)
					
					# init sigma_2
					n_row = Int64(GrB_Matrix_nrows(M_1));
					n_col = Int64(GrB_Matrix_nrows(M_2));
					I1 = [i for i=1:n_row-1]
					J1 = [j for j=1:n_col-1]
					x = [1 for i in 1:n_row]
					sigma_2_notbinary = GrB_Matrix(ZeroBasedIndex.(I1),ZeroBasedIndex.(J1),x );


					# compute sigma_2
					
					function compute_s2()
						GrB_mxm(sigma_2_notbinary, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, M_1, M_2, desc);
						#factor = 0.5;
						#sigma_2 = normalize_sigma(sigma_2_notbinary, factor); # sigma_2 have zeros in x
						sigma_2 = intdivbyn_sigma(sigma_2_notbinary,2);
					end
					
					@info("M1*M2'÷2 ")
					
					t2 = Base.@elapsed @btime $compute_s2()

					
					
					
					# init sigma_3
					n_row = Int64(GrB_Matrix_nrows(M_2));
					n_col = Int64(GrB_Matrix_nrows(M_3));
					I1 = [i for i=1:n_row-1]
					J1 = [j for j=1:n_col-1]
					x = [1 for i in 1:n_row]
					sigma_3_notbinary = GrB_Matrix(ZeroBasedIndex.(I1),ZeroBasedIndex.(J1),x );


					# compute sigma_3
					
					function compute_s3()
						GrB_mxm(sigma_3_notbinary, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, M_2, M_3, desc);
						factor = 1;
						#sigma_3 = normalize_sigma_float(sigma_3_notbinary, factor); 
						#sigma_3 = normalize_sigma(sigma_3, factor); # sigma_3 have zeros in x
						sigma_3 = floatdivbyn_sigma(sigma_3_notbinary,4)
						sigma_3_2 = intdivbyn_sigma(sigma_3, 1)
					end
					
					
					@info("(M2*M3'/4)÷1")
					t3 = Base.@elapsed @btime $compute_s3();
				
				elseif K == K_gpu_sparse
					try
						@info("M0*M1'")
						t1 = Base.@elapsed @btime CUSPARSE.gemm('N','T',$M_0,$M_1,'O','O','O');
					catch e
						println("Dim too big")
					end 
					
					try
						@info("M1*M2'÷2 ")
						t2 = Base.@elapsed @btime collect((CUSPARSE.gemm('N','T',$M_1,$M_2,'O','O','O'))) .÷ 2; #divisione su cpu
					catch e
						println("Dim too big")
					end
					
					try	
						@info("(M1*M3'/4)÷1")
						t3 = Base.@elapsed @btime (collect(CUSPARSE.gemm('N','T',$M_2,$M_3,'O','O','O')) ./ 4) .÷ 1;
					catch e
						println("Dim too big")
					end 
					
					print("\n")
				
					
				else
					@info("M0*M1'")
					t1 = Base.@elapsed @btime $M_0 * $M_1';
					@info("M1*M2'÷2 ")
					t2 = Base.@elapsed @btime ($M_1 * $M_2') .÷ 2;
					@info("(M1*M3'/4)÷1")
					t3 = Base.@elapsed @btime (($M_2 * $M_3')  ./ 4) .÷ 1;	
					print("\n")
				end

				if K == K_cpu
					push!(jltimes,t1)
					push!(jltimes,t2)
					push!(jltimes,t3)
				
				elseif K == K_gpu
					push!(cutimes,t1)
					push!(cutimes,t2)
					push!(cutimes,t3)
				
				elseif K == K_gpu_sparse
					push!(cusparsetimes,t1)
					push!(cusparsetimes,t2)
					push!(cusparsetimes,t3)
				else 
					push!(graphtimes,t1)
					push!(graphtimes,t2)
					push!(graphtimes,t3)
				end
			end

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


function delete_0_elements_in_sigma(I,J,X)

	i1,j1,x1 = [],[],Int64[]

	for (i,j,x) in zip(I,J,X)
		#print(i,j,x)
		if x > 0
			push!(i1,i)
			push!(j1,j)
			push!(x1,x)
		else
		end
	end
	return i1,j1,x1
end


function delete_0_elements_in_sigma_float(I,J,X)

	i1,j1,x1 = [],[],Float64[]

	for (i,j,x) in zip(I,J,X)
		#print(i,j,x)
		if x > 0
			push!(i1,i)
			push!(j1,j)
			push!(x1,x)
		else
		end
	end
	return i1,j1,x1
end


function normalize_sigma(sigma2norm, factor)

	n_row = Int64(GrB_Matrix_nrows(sigma2norm))
	n_col = Int64(GrB_Matrix_ncols(sigma2norm))
	I1 = OneBasedIndex.([i for i=1:n_row])
	J1 = OneBasedIndex.([j for j=1:n_col])

	OUT = GrB_Matrix{Int64}()
	GrB_Matrix_new(OUT, GrB_INT64, n_row, n_col)
	GrB_Matrix_extract(OUT, GrB_NULL, GrB_NULL, sigma2norm, I1, n_row, J1, n_col, GrB_NULL)

	I, J, X = GrB_Matrix_extractTuples(sigma2norm)
	X_divisor = [factor for i=1:length(X)]
	sigma_divisor = GrB_Matrix(I, J, X_divisor)

	sigma0 = GrB_Matrix{Int64}()
	GrB_Matrix_new(sigma0, GrB_INT64, n_row, n_col)
	GrB_eWiseMult_Matrix_Semiring(sigma0, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_FP64, sigma2norm, sigma_divisor, GrB_NULL)


	#@GxB_fprint(sigma0,GxB_COMPLETE)
	#delete 0-elements in sigma
	n_row = Int64(GrB_Matrix_nrows(sigma0))
	n_col = Int64(GrB_Matrix_ncols(sigma0))
	I1 = OneBasedIndex.([i for i=1:n_row])
	J1 = OneBasedIndex.([j for j=1:n_col])

	OUT_sigma0 = GrB_Matrix{Int64}()
	GrB_Matrix_new(OUT_sigma0, GrB_INT64, n_row, n_col)
	GrB_Matrix_extract(OUT_sigma0, GrB_NULL, GrB_NULL, sigma0, I1, n_row, J1, n_col, GrB_NULL)

	I, J, X = GrB_Matrix_extractTuples(sigma0)

	i1,j1,x1 = delete_0_elements_in_sigma(ZeroBasedIndex2Int(I), ZeroBasedIndex2Int(J), X)
	#print(I, J, X )
	sigma = GrB_Matrix(OneBasedIndex.(i1), OneBasedIndex.(j1), x1)

	return sigma

end

function normalize_sigma_float(sigma2norm, factor)

	n_row = Int64(GrB_Matrix_nrows(sigma2norm))
	n_col = Int64(GrB_Matrix_ncols(sigma2norm))
	I1 = OneBasedIndex.([i for i=1:n_row])
	J1 = OneBasedIndex.([j for j=1:n_col])

	OUT = GrB_Matrix{Int64}()
	GrB_Matrix_new(OUT, GrB_INT64, n_row, n_col)
	GrB_Matrix_extract(OUT, GrB_NULL, GrB_NULL, sigma2norm, I1, n_row, J1, n_col, GrB_NULL)

	I, J, X = GrB_Matrix_extractTuples(sigma2norm)
	X_divisor = [factor for i=1:length(X)]
	sigma_divisor = GrB_Matrix(I, J, X_divisor)

	sigma0 = GrB_Matrix{Float64}()
	GrB_Matrix_new(sigma0, GrB_FP64, n_row, n_col)
	GrB_eWiseMult_Matrix_Semiring(sigma0, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_FP64, sigma2norm, sigma_divisor, GrB_NULL)


	#@GxB_fprint(sigma0,GxB_COMPLETE)
	#delete 0-elements in sigma
	n_row = Int64(GrB_Matrix_nrows(sigma0))
	n_col = Int64(GrB_Matrix_ncols(sigma0))
	I1 = OneBasedIndex.([i for i=1:n_row])
	J1 = OneBasedIndex.([j for j=1:n_col])

	OUT_sigma0 = GrB_Matrix{Int64}()
	GrB_Matrix_new(OUT_sigma0, GrB_INT64, n_row, n_col)
	GrB_Matrix_extract(OUT_sigma0, GrB_NULL, GrB_NULL, sigma0, I1, n_row, J1, n_col, GrB_NULL)

	I, J, X = GrB_Matrix_extractTuples(sigma0)

	i1,j1,x1 = delete_0_elements_in_sigma_float(ZeroBasedIndex2Int(I), ZeroBasedIndex2Int(J), X)
	#print(I, J, X )
	sigma = GrB_Matrix(OneBasedIndex.(i1), OneBasedIndex.(j1), x1)

	return sigma

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













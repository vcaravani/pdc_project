using LinearAlgebraicRepresentation, SparseArrays, DataStructures
Lar = LinearAlgebraicRepresentation

using CuArrays,CuArrays.CUSPARSE

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



S2 = sum(sigma_3_cu,dims=2)
S2 = collect(S2)

inner = [k for k=1:length(S2) if S2[k]==2]
outer = setdiff(collect(1:length(FV)), inner)


GL.VIEW([ GL.GLGrid(V,EV,GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[inner],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[outer],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);


cutimes, jltimes, cusparsetimes = Float64[], Float64[], Float64[]

function benchmark(n,m,k)

	V,CV = Lar.cuboidGrid([n,m,k])
	VV = [[v] for v=1:size(V,2)]
	EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))
	FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))
	
	t1,t2,t3 = 0.0, 0.0, 0.0 #where 0.0 = not defined
	
    K_cpu, K_gpu, K_gpu_sparse = K_f, K_cuda, K_sparse

			for K in (K_cpu,K_gpu,K_gpu_sparse)

				M_0 = K(VV)
				M_1 = K(EV)
				M_2 = K(FV)
				M_3 = K(CV)

				println(typeof(M_0))
				
				if K == K_gpu_sparse
					try
						t1 = Base.@elapsed @btime CUSPARSE.gemm('N','T',$M_0,$M_1,'O','O','O');
					catch e
						println("Dim to big")
					end 
					
					try
						t2 = Base.@elapsed @btime cu(collect((CUSPARSE.gemm('N','T',$M_1,$M_2,'O','O','O')))) .÷ 2;
					catch e
						println("Dim to big")
					end
					
					try	
						t3 = Base.@elapsed @btime collect(CUSPARSE.gemm('N','T',$M_2,$M_3,'O','O','O')) .÷ 4;
					catch e
						println("Dim to big")
					end 
				else
					t1 = Base.@elapsed @btime $M_0 * $M_1'
					t2 = Base.@elapsed @btime ($M_1 * $M_2') .÷ 2
					t3 = Base.@elapsed @btime($M_2 * $M_3')  .÷ 4
				end

				if K == K_cpu
					push!(jltimes,t1)
					push!(jltimes,t2)
					push!(jltimes,t3)
				
				elseif K == K_gpu
					push!(cutimes,t1)
					push!(cutimes,t2)
					push!(cutimes,t3)
				else
					push!(cusparsetimes,t1)
					push!(cusparsetimes,t2)
					push!(cusparsetimes,t3)

				end
			end

end
















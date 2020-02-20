using LinearAlgebraicRepresentation, SparseArrays
Lar = LinearAlgebraicRepresentation

using CuArrays,CuArrays.CUSPARSE

using BenchmarkTools


function random3cells(shape,npoints)
	pointcloud = rand(3,npoints).*shape
	grid = DataStructures.DefaultDict{Array{Int,1},Int}(0)

	for k = 1:size(pointcloud,2)
		v = map(Int∘trunc,pointcloud[:,k])
		if grid[v] == 0 # do not exists
			grid[v] = 1
		else
			grid[v] += 1
		end
	end

	out = Array{Lar.Struct,1}()
	for (k,v) in grid
		V = k .+ [
		 0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0;
		 0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0;
		 0.0  1.0  0.0  1.0  0.0  1.0  0.0  1.0]
		cell = (V,[[1,2,3,4,5,6,7,8]])
		push!(out, Lar.Struct([cell]))
	end
	out = Lar.Struct( out )
	V,CV = Lar.struct2lar(out)
end



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


function K_device( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = Float32[1 for k=1:length(I)]
	CV_sparse = sparse(I,J,Vals)
	d_CV = CuSparseMatrixCSC(CV_sparse)

	return d_CV
end

function K( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return sparse(I,J,Vals)
end

function K_cu( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return cu(sparse(I,J,Vals))
end

n = 3;
m = 2;
k = 1;

V,CV = Lar.cuboidGrid([n,m,k])

#V,CV = random3cells([40,20,	10],4_000)


VV = [[v] for v=1:size(V,2)]
EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))
FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))


M_0 = K(VV);
M_1 = K(EV);
M_2 = K(FV);
M_3 = K(CV);

sigma_1 =  M_0 * M_1';
sigma_2 = M_1 * M_2' .÷ 2;
#sigma_3 = M_2 * M_3' .÷ 4;
s = sum(M_2,dims=2)
sigma_3 = (M_2 * M_3')
sigma_3 = sigma_3 ./	s
sigma_3 = ∂_3 .÷ 1



M_0_cu = cu(M_0);
M_1_cu = cu(M_1);
M_2_cu = cu(M_2);
M_3_cu = cu(M_3);

#CuArrays.allowscalar(false)

sigma_1_cu = M_0_cu * M_1_cu';
sigma_2_cu = M_1_cu * M_2_cu' .÷ 2;
sigma_3_cu = (M_2_cu * M_3_cu') ./ 4;
sigma_3_cu = sigma_3_cu .÷ 1;

using Test
@test Matrix(sigma_1) == collect(sigma_1_cu)
@test Matrix(sigma_2) == collect(sigma_2_cu)
@test Matrix(sigma_3) == collect(sigma_3_cu)

cutimes, jltimes = Float64[], Float64[]

function benchmark(n,m,k)

	V,CV = Lar.cuboidGrid([n,m,k])
	VV = [[v] for v=1:size(V,2)]
	EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))
	FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))

    K_cpu, K_gpu = K, K_cu

			for K in (K_cpu,K_gpu)

				M_0 = K(VV)
				M_1 = K(EV)
				M_2 = K(FV)
				M_3 = K(CV)

				println(typeof(M_0))

				t1 = Base.@elapsed @btime $M_0 * $M_1'
				t2 = Base.@elapsed @btime ($M_1 * $M_2') .÷ 2
				t3 = Base.@elapsed @btime($M_2 * $M_3')  .÷ 4

				if K == K_cpu
					push!(jltimes,t1)
					push!(jltimes,t2)
					push!(jltimes,t3)
				else
					push!(cutimes,t1)
					push!(cutimes,t2)
					push!(cutimes,t3)
				end

			end

end

# https://nextjournal.com/sdanisch/julia-gpu-programming per levare gli if take a look!!!!



s1 = collect(sigma_1_cu)
s2 = collect(sigma_2_cu)
s3 = collect(sigma_3_cu)


S2 = sum(s3,dims=2)
S2 = collect(S2)

inner = [k for k=1:length(S2) if S2[k]==2]
outer = setdiff(collect(1:length(FV)), inner)


GL.VIEW([ GL.GLGrid(V,EV,GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[inner],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[outer],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

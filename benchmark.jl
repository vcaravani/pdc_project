using BenchmarkTools

using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation

using SparseArrays, CuArrays

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

				print(string(typeof(M_0))*"\n")

				t1 = Base.@elapsed @btime $M_0 * $M_1'
				t2 = Base.@elapsed @btime ($M_1 * $M_2') .÷ 2
				t3_1 = Base.@elapsed @btime($M_2 * $M_3')  ./ 4
				t3_2 = Base.@elapsed @btime($M_2 * $M_3')  .÷ 1

				if K == K_cpu
					push!(jltimes,t1)
					push!(jltimes,t2)
					push!(jltimes,t3_1+t3_2)
				else
					push!(cutimes,t1)
					push!(cutimes,t2)
					push!(cutimes,t3_1+t3_2)
				end

			end

end

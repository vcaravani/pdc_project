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



cutimes, jltimes, nvertex = Float64[], Float64[], Int64[]

function benchmark_jl_vs_cu(n,m,kstep_size,iteration)

	sz = step_size
	max_it = iteration


	for i in 1:sz:max_it
		#print([n*i,m*i,k*i])
		V,CV = Lar.cuboidGrid([n*i,m*i,k])

		print("n_vertex: "*string(size(V)[2]), ([n*i,m*i,k]))
		push!(nvertex, size(V)[2])

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

					t1 = @belapsed $M_0 * $M_1'
					t2 = @belapsed ($M_1 * $M_2') #.÷ 2
					t3 = @belapsed($M_2 * $M_3')  #./ 4
					#t3_2 = @belapsed $M_2 * $M_3')  #.÷ 1

					if K == K_cpu
						#push!(jltimes,t1)
						#push!(jltimes,t2)
						push!(jltimes,t1+t2+t3)
					else
						#push!(cutimes,t1)
						#push!(cutimes,t2)
						push!(cutimes,t1+t2+t3)
					end

				end
	end

end


speedup = jltimes ./ cutimes
max_range = length(speedup)

s = 5
it = 30
benchmark_jl_vs_gblas(8,4,10,s,it)

pop!(nvertex)
ranges = 1:max_range
x = [90,9000,50000,150000,300000,500000 ]
x = nvertex
using Plots; plotly()
#x = repeat(ranges, inner = 1)
Plots.scatter(
  log.(x), [speedup, fill(1.0, length(speedup))],
  label = ["gblas" "sparse_arrays"], markersize = 4, markerstrokewidth = 1,
  legend = :right, xlabel = "vertex", ylabel = "speedup"
)

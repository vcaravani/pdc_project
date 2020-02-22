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




function benchmark_jl_vs_cu(nmk)


	for i in 1:length(nmk)
		#print([n*i,m*i,k*i])
		V,CV = Lar.cuboidGrid(nmk[i])

		print("n_vertex: "*string(size(V)[2]), (nmk[i]))
		push!(nvertex, size(V)[2])

		VV = [[v] for v=1:size(V,2)]
		EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))
		FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))

	    K_cpu, K_gpu = K, K_cu

				for K in (K_cpu,K_gpu)

					M_0 = K(VV)
					M_1 = K(EV)
					M_2 = K(FV)
					M_3 = K(CV)

					#print(string(typeof(M_0))*"\n")
					try
						t1 = @belapsed $M_0 * $M_1'
					catch e
						t1 = 10

					try
						t2 = @belapsed ($M_1 * $M_2') #.÷ 2ù
					catch e
						t2 = 10
					try
						t3 = @belapsed($M_2 * $M_3')  #./ 4
						#t3_2 = @belapsed $M_2 * $M_3')  #.÷ 1
					catch e
						t3 = 10

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


cutimes, jltimes, nvertex = Float64[], Float64[], Int64[]

#nmk = [[10,5,2],[40,20,10],[80,40,20],[100,50,25],[200,100,50],[300,150,50]] #outofmemory in 40,20,10 su gtx nvidia 1060
nmk = [[3,2,1],[10,5,1],[20,10,1], [20,10,1], [30,10,1], [40,20,1], [60,30,1]]
benchmark_jl_vs_cu(nmk)

speedup = jltimes ./ cutimes
x = nvertex

print(x) # per la tesla
print(speedup) # per la tesla

using Plots; plotly()
#x = repeat(ranges, inner = 1)
Plots.scatter(
  x, [speedup, fill(1.0, length(speedup))],
  label = ["cuarrays" "sparse_arrays"], markersize = 4, markerstrokewidth = 1,
  legend = :right, xlabel = "vertex", ylabel = "speedup"
)

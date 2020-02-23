using BenchmarkTools

using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation

using SparseArrays, CuArrays.CUSPARSE

function K( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return sparse(I,J,Vals)
end

function K_cusparse( CV )

	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = Float32[1 for k=1:length(I)]
	M = sparse(I,J,Vals)
	d_M = CuSparseMatrixCSR(M)

	return d_M
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




function benchmark_jl_vs_cusparse(nmk)


	for i in 1:length(nmk)
		#print([n*i,m*i,k*i])
		V,CV = Lar.cuboidGrid(nmk[i])

		print("n_vertex: "*string(size(V)[2]), (nmk[i]))
		push!(nvertex, size(V)[2])

		VV = [[v] for v=1:size(V,2)]
		EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))
		FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))

		t1,t2,t3 = 0.0,0.0,0.0


	    K_cpu, K_gpu = K, K_cusparse

				for K in (K_cpu,K_gpu)

					M_0 = K(VV)
					M_1 = K(EV)
					M_2 = K(FV)
					M_3 = K(CV)


					if K == K_cpu

						#print(string(typeof(M_0))*"\n")

						t1 = @belapsed $M_0 * $M_1'
						t2 = @belapsed ($M_1 * $M_2') #.÷ 2
						t3 = @belapsed($M_2 * $M_3')  #./ 4
						#t3_2 = @belapsed $M_2 * $M_3')  #.÷ 1

						#push!(jltimes,t1)
						#push!(jltimes,t2)
						push!(jltimes,t1+t2+t3)
					else

				
						try
						t1 = @belapsed CUSPARSE.gemm('N','T',$M_0,$M_1,'O','O','O');
						catch e
							print("GPU out of memory: ", e)
							t1 = 10
						end

						try
							t2 = @belapsed CUSPARSE.gemm('N','T',$M_1,$M_2,'O','O','O');
						catch e
							t2 = 10
							print("GPU out of memory: ", e)
						end
						try
							t3 = @belapsed CUSPARSE.gemm('N','T',$M_2,$M_3,'O','O','O');
						catch e
							t3 = 10
							print("GPU out of memory: ", e)
						end
						#push!(cutimes,t1)
						#push!(cutimes,t2)
						push!(cutimes,t1+t2+t3)
					end

				end
	end

end


cutimes, jltimes, nvertex = Float64[], Float64[], Int64[]

nmk = [[10,5,2],[40,20,10],[80,40,20],[100,50,25],[200,100,50],[300,150,50]]

benchmark_jl_vs_cusparse(nmk)

speedup = jltimes ./ cutimes
x = nvertex


using Plots; plotly()
#x = repeat(ranges, inner = 1)
Plots.scatter(
  x, [speedup, fill(1.0, length(speedup))],
  label = ["cusparse" "sparse_arrays"], markersize = 4, markerstrokewidth = 1,
  legend = :right, xlabel = "vertex", ylabel = "speedup"
)

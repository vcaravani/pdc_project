using BenchmarkTools

using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation

using SparseArrays, CuArrays

using SuiteSparseGraphBLAS, GraphBLASInterface, Test
GrB_init(GrB_NONBLOCKING)

gblastimes, jltimes, nvertex = Float64[], Float64[], Int64[]

function benchmark_jl_vs_gblas(n,m,k,step_size,iteration)

	sz = step_size
	max_it = iteration


	for i in 1:sz:max_it
		#print([n*i,m*i,k*i])
		V,CV = Lar.cuboidGrid([n*i,m*i,k])

		print("n_vertex: "*string(size(V)[2]), ([n*i,m*i,k]))
		push!(nvertex, size(V)[2])
		print("\n\n")
		VV = [[v] for v=1:size(V,2)]
		EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))
		FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))

	    K_cpu, K_gblas = K, K_GBLAS

				for K in (K_cpu,K_gblas)



					if K == K_cpu

						M_0 = K(VV)
						M_1 = K(EV)
						M_2 = K(FV)
						M_3 = K(CV)



						t1 = @belapsed $M_0 * $M_1'
						t2 = @belapsed ($M_1 * $M_2')
						# .÷ 2
						t3 = @belapsed ($M_2 * $M_3')
						#  ./ 4) .÷ 1


						push!(jltimes,t1+t2+t3)
						#push!(jltimes,t2)
						#push!(jltimes,t3)


					elseif K == K_gblas

						M_0 = K(VV)
						M_1 = K(EV)
						M_2 = K(FV)
						M_3 = K(CV)
						# init GrB_Descriptor for mxm
						desc = init_descriptor_mxm();

						# compute ∂_1
						∂_1 = init_∂(M_0,M_1);
						t1 = @belapsed GrB_mxm($∂_1, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, $M_0, $M_1, $desc)
						t1 =  t1

						# compute ∂_2
						∂_2 = init_∂(M_1,M_2);
						t2 = @belapsed GrB_mxm($∂_2, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, $M_1, $M_2, $desc)
						#∂_2 = intdivbyn_sigma(∂_2,$integer_division_by_2);

						# compute ∂_3
						∂_3 = init_∂(M_2,M_3);
						t3 = @belapsed GrB_mxm($∂_3, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, $M_2, $M_3, $desc)
						#∂_3 = floatdivbyn_sigma(∂_3,$float_division_by_4)
						#∂_3 = intdivbyn_sigma(∂_3, $integer_division_by_1)



						tot_time_gblas = t1+t2+t3


						push!(gblastimes,tot_time_gblas)
						#push!(gblastimes,t2)
						#push!(gblastimes,t3)
					end

				end
			end
end


s = 5
it = 30
benchmark_jl_vs_gblas(8,4,10,s,it)

speedup = jltimes ./ gblastimes
x = nvertex

using Plots; plotly()
#x = repeat(ranges, inner = 1)
Plots.scatter(
  log2.(x), [speedup, fill(1.0, length(speedup))],
  label = ["gblas" "sparse_arrays"], markersize = 4, markerstrokewidth = 1,
  legend = :right, xlabel = "vertex", ylabel = "speedup"
)

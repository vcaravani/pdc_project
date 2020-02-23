using BenchmarkTools

using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation

using SparseArrays, CuArrays

using SuiteSparseGraphBLAS, GraphBLASInterface, Test
GrB_init(GrB_NONBLOCKING)



function benchmark_jl_vs_gblas(nmk)


	for i in 1:length(nmk)
		V,CV = Lar.cuboidGrid(nmk[i])

		print("n_vertex: "*string(size(V)[2]), (nmk[i]))
		push!(nvertex, size(V)[2])
		print("\n\n")
		VV = [[v] for v=1:size(V,2)]
		EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))
		FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))

		t1,t2,t3 = 0.0,0.0,0.0


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




gblastimes, jltimes, nvertex = Float64[], Float64[], Int64[]

nmk = [[10,5,2],[40,20,10],[80,40,20],[100,50,25],[200,100,50],[300,150,50]]

benchmark_jl_vs_gblas(nmk)

speedup = jltimes ./ gblastimes
x = nvertex

using Plots; plotly()
#x = repeat(ranges, inner = 1)
Plots.scatter(
  x, [speedup, fill(1.0, length(speedup))],
  label = ["gblas" "sparse_arrays"], markersize = 4, markerstrokewidth = 1,
  legend = :right, xlabel = "vertex", ylabel = "speedup"
)

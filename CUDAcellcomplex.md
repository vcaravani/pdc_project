# Applying GraphBLAS standard to Chain Complexes (currently with JuliaGPU.jl)

#= Transformation from Arrays of arrays of integer indices to sparse GraphBLAS matrices =#

using CuArrays
using CuArrays.CUSPARSE


### Generation of the B-rep of topology a unit cube using LAR 

#= A Linear Algebraic Representation (LAR) of a unit cube may be generated in Julia using the LinearAlgebraicRepresentation.jl package. In particular we generate a 3D grid of unit cubes with a single element: =#

	using LinearAlgebraicRepresentation,SparseArrays
	Lar = LinearAlgebraicRepresentation

### LAR cellular complexes generation

	V,CV = Lar.cuboidGrid([3,2,1])
	CV = convert(Array{Array{Int32,1},1},CV)
	
	function CV2FV( v::Array )
		faces = Array{Int32,1}[
			[v[1], v[2], v[3], v[4]], [v[5], v[6], v[7], v[8]],
			[v[1], v[2], v[5], v[6]],	[v[3], v[4], v[7], v[8]], 
			[v[1], v[3], v[5], v[7]], [v[2], v[4], v[6], v[8]]]
	end	

	function CV2EV( v::Array )
		edges = Array{Int32,1}[
			[v[1], v[2]], [v[3], v[4]], [v[5], v[6]], [v[7], v[8]], [v[1], v[3]], [v[2], v[4]],
			[v[5], v[7]], [v[6], v[8]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]], [v[4], v[8]]]
	end	
	
	VV = [[v] for v=1:size(V,2)]
	FV = collect(Set(vcat(map(CV2FV,CV)...)))
	EV = collect(Set(vcat(map(CV2EV,CV)...)))


### Generation of "characteristic matrices" of cellular complexes

	function K( CV )
		I = vcat( [ Int32[k for h in CV[k]] for k=1:length(CV) ]...)
		J = vcat(CV...)
		Vals = Int8[1 for k=1:length(I)]
		return SparseArrays.sparse(I,J,Vals)
	end
	
	function cuK( CV )
		cuI = CuArray( vcat( [ Int32[k for h in CV[k]] for k=1:length(CV) ]...) )
		cuJ = CuArray( vcat(CV...) )
		cuVals = CuArray( Int8[1 for k=1:length(cuI)] )
		return CuArrays.CUSPARSE.sparse(cuI,cuJ,cuVals)
	end


	k_0 = K(VV)
	k_1 = K(EV)
	k_2 = K(FV)
	k_3 = K(CV)
	
	julia> @btime cuk_3 = cuK(CV);
  2.609 ms (961 allocations: 39.72 KiB)

	julia> @btime k_3 = K(CV);
  1.425 μs (26 allocations: 4.70 KiB)

	
	
	
### Generation of "boundary operators" of cellular complexes

	# sparse arrays
	∂_1 = k_0 * k_1'  
	∂_2 = (k_1 * k_2') .÷ 2 # sum(k_1,dims=2) 	
	∂_3 = (k_2 * k_3') ./	sum(k_2,dims=2) 	#	.÷ sum(k_2,dims=2) 

	# dense arrays
	B_1 = Matrix(∂_1)
	B_2 = Matrix(∂_2)
	B_3 = Matrix(∂_3) .÷ 1

### Tests
	S1 = sum(B_3,dims=1)
	S2 = sum(B_3,dims=2)

### 
	b2 = B_3 .* ones(size(VV,2),1)
	sum(b2,dims=1)
	
### interior/exterior 2-cells
	int = [k for k=1:length(S2) if S2[k]==2]
	ext = setdiff(collect(1:length(FV)), int)
	
### number of interior 2-cells
	sum([1 for k=1:length(S2) if S2[k]==2])

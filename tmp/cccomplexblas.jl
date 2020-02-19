|
using CuArrays
using CuArrays.CUSPARSE
using DataStructures
using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation
using SparseArrays
using CUDAnative
using BenchmarkTools, Test

# ===================================================================================

m = 25
n = 35
k = 10
elty = Float32

# ===================================================================================
function CV2FV( v::Array )
	faces = [
		Int32[v[1], v[2], v[3], v[4]], [v[5], v[6], v[7], v[8]],
		Int32[v[1], v[2], v[5], v[6]],	[v[3], v[4], v[7], v[8]], 
		Int32[v[1], v[3], v[5], v[7]], [v[2], v[4], v[6], v[8]]]
end	

function CV2EV( v::Array )
	edges = [
		Int32[v[1], v[2]], [v[3], v[4]], [v[5], v[6]], [v[7], v[8]], [v[1], v[3]], [v[2], v[4]],
		Int32[v[5], v[7]], [v[6], v[8]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]], [v[4], v[8]]]
end	

function K( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...);
	J = vcat(CV...);
	Vals = Float32[1 for k=1:length(I)];
	return sparse(I,J,Vals);
end


V,CV = Lar.cuboidGrid([64,64,64]);
CV = convert( Array{Array{Int32,1},1}, CV );
V = convert(Array{Float32,2}, V );
VV = [Int32[v] for v=1:size(V,2)];
FV = collect(Set(vcat(map(CV2FV,CV)...)));
EV = collect(Set(vcat(map(CV2EV,CV)...)));
M_0 = K(VV); M_1 = K(EV); M_2 = K(FV); M_3 = K(CV);

m,k = size(M_2)
n,k = size(M_3)
∂_2 = M_2 * M_3';
dM_2 = CuSparseMatrixCSR(M_2);
dM_3 = CuSparseMatrixCSR(M_3);
d∂_2 = CUSPARSE.gemm('N','T',dM_2,dM_3,'O','O','O');
@test ∂_2 ≈ collect(d∂_2)

x = ones(elty,k);
d_x = CuArray(x);
d_inverse_sum = 1.0f0 ./ CUSPARSE.mv('N',dM_2,d_x,'O');

# next instructions errored:
typeof(d∂_2), typeof(d_inverse_sum)
size(d∂_2), size(d_inverse_sum)

typeof(sparse(rand(10,5))), typeof(fill(10, 10))
size(sparse(rand(10,5))), size(fill(10, 10))

# d∂_2 .* d_inverse_sum
# d∂_2 = CUSPARSE.gemm('N','N',d∂_2,d_inverse_sum,'O','O','O');

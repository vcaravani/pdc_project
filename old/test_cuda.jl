using LinearAlgebraicRepresentation, SparseArrays, DataStructures
Lar = LinearAlgebraicRepresentation

using ViewerGL
GL = ViewerGL

function CV2FV( v::Array{Int32} )
	faces = [
		Int32[v[1], v[2], v[3], v[4]], [v[5], v[6], v[7], v[8]],
		Int32[v[1], v[2], v[5], v[6]],	[v[3], v[4], v[7], v[8]],
		Int32[v[1], v[3], v[5], v[7]], [v[2], v[4], v[6], v[8]]]
end

function CV2EV( v::Array{Int32} )
	edges = [
		Int32[v[1], v[2]], [v[3], v[4]], [v[5], v[6]], [v[7], v[8]], [v[1], v[3]], [v[2], v[4]],
		Int32[v[5], v[7]], [v[6], v[8]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]], [v[4], v[8]]]
end

function K_cuda( CV, trans = false)
	I = vcat( Array{Int32,1}[ Int32[k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]

	if trans == false
		m = CuArrays.CUSPARSE.sparse(I,J,Vals)
	else
		m = CuArrays.CUSPARSE.sparse(J,I,Vals)
	end
	ms = convert(SparseMatrixCSC{Float32,Int64}, m)
	d_M = CuSparseMatrixCSC(ms);

	return  d_M
end

function K( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return SparseArrays.sparse(I,J,Vals)
end

V,CV = Lar.cuboidGrid([3,2,1])

VV = [Int32[v] for v=1:size(V,2)]
FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))
EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))

#=
M_0 = K(VV)
M_1 = K(EV)
M_2 = K(FV)
M_3 = K(CV)
=#
using CuArrays.CUSPARSE

M_0_cuda = K_cuda(VV);
M_1_cuda = K_cuda(EV);
M_2_cuda = K_cuda(FV);
M_3_cuda = K_cuda(CV);

#=
∂_1 = M_0 * M_1'
∂_2 = (M_1 * M_2') .÷ 2 #	.÷ sum(M_1,dims=2)
s = sum(M_2,dims=2)
∂_3 = (M_2 * M_3')
∂_3 = ∂_3 ./	s
∂_3 = ∂_3 .÷ 1	#	.÷ sum(M_2,dims=2)
=#

M_1_cuda_trans = K_cuda(EV,true);
∂_1 = M_0_cuda * M_1_cuda;
∂_2 = (M_1 * M_2') .÷ 2 #	.÷ sum(M_1,dims=2)
s = sum(M_2,dims=2)
∂_3 = (M_2 * M_3')
∂_3 = ∂_3 ./	s
∂_3 = ∂_3 .÷ 1	#	.÷ sum(M_2,dims=2)

S2 = sum(∂_3,dims=2)
inner = [k for k=1:length(S2) if S2[k]==2]
outer = setdiff(collect(1:length(FV)), inner)







GL.VIEW([ GL.GLGrid(V,EV,GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[inner],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[outer],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

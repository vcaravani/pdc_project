using CuArrays
using CuArrays.CUSPARSE
#=
i = Int32[1,2]
j = Int32[2,1]
v = Float32[10,20]

ms = CuArrays.CUSPARSE.sparse(i,j,v)
a = CuSparseMatrixCSC(ms);

i2 = Int32[1,2]
j2= Int32[2,1]
v2 = Float32[100,200]

ms2 = CuArrays.CUSPARSE.sparse(i2,j2,v2)
b = CuSparseMatrixCSC(ms2);


ms3 = CuArrays.CUSPARSE.sparse(Float32.(rand(2,2)))
c = CuSparseMatrixCSC(ms3);
=#

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

using LinearAlgebraicRepresentation, SparseArrays, DataStructures
Lar = LinearAlgebraicRepresentation

using ViewerGL
GL = ViewerGL

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

V,CV = Lar.cuboidGrid([3,2,1])

VV = [Int32[v] for v=1:size(V,2)]
FV = convert(Array{Array{Int32,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))
EV = convert(Array{Array{Int32,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))

M_0_cuda = K_cuda(VV);
M_1_cuda = K_cuda(EV);
M_2_cuda = K_cuda(FV);
M_3_cuda = K_cuda(CV);


#=
function mm(transa::SparseChar,
            A::CuSparseMatrix{$elty},
            B::CuMatrix{$elty},
            beta::$elty,
            C::CuMatrix{$elty},
            index::SparseChar)
    mm(transa,one($elty),A,B,beta,C,index)
end

function mm(transa::SparseChar,
			A::CuSparseMatrix{$elty},
			B::CuMatrix{$elty},
			C::CuMatrix{$elty},
			index::SparseChar)
	mm(transa,one($elty),A,B,one($elty),C,index)
end

proviamo a fare questa
=#

#aa = cu((rand(2,2)))
#[2 1,0 1]
#[1 0, 0 1]
#bb = cu((rand(2,2)))
#c_s = CuArrays.CUSPARSE.sparse(Float32.(rand(2,2)))
#c = CuSparseMatrixCSC(c_s);
#CUSPARSE.mm('N',c,aa,bb,'O');

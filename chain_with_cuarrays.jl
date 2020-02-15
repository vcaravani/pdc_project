using LinearAlgebraicRepresentation, SparseArrays, DataStructures
Lar = LinearAlgebraicRepresentation

using CuArrays,CuArrays.CUSPARSE

using ViewerGL
GL = ViewerGL


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


function K_device( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = Float32[1 for k=1:length(I)]
	CV_sparse = sparse(I,J,Vals)
	d_CV = CuSparseMatrixCSC(CV_sparse)

	return d_CV
end

function generate_chain_complex_cu()

	V,CV = Lar.cuboidGrid([3,2,1])

	VV = [[v] for v=1:size(V,2)]
	EV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2EV,CV)...))))
	FV = convert(Array{Array{Int64,1},1}, collect(Set(vcat(map(CV2FV,CV)...))))

	M_0 = K_device(VV);
	M_1 = K_device(EV);
	M_2 = K_device(FV);
	M_3 = K_device(CV);

	M_0_cu = cu(M_0)
	M_1_cu = cu(M_1)
	M_2_cu = cu(M_2)
	M_3_cu = cu(M_3)


	sigma_1_cu = M_0_cu * M_1_cu'
	sigma_2_cu = M_1_cu * M_2_cu' .÷ 2
	sigma_3_cu = (M_2_cu * M_3_cu') .÷ 4

	return sigma_3_cu, V, EV, FV

end

S2 = sum(sigma_3_cu,dims=2)
S2 = collect(S2)

inner = [k for k=1:length(S2) if S2[k]==2]
outer = setdiff(collect(1:length(FV)), inner)


GL.VIEW([ GL.GLGrid(V,EV,GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[inner],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

GL.VIEW([ GL.GLGrid(V,FV[outer],GL.Point4d(1,1,1,1))
          GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

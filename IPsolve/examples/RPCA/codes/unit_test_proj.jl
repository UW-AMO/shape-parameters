#======= Unit Test for projκ =======#
include("./utility.jl")
include("./RPCA_Solver.jl")
include("../../../1Dmin/rminbnd.jl")

# generate huber samples
m,n,κ = 100,100,0.5
R = zeros(m,n)
for I in eachindex(R)
    R[I] = sampleH(κ)
end

# κ̄ = projκ(R,0.5)
# @show κ̄

f   = κ -> κfun(R,κ)
κ̄,f̄ = rminbnd(f,0.0,1.0,κ)
@show κ̄
#======= Synethic Data Test Solver =======#
include("./utility.jl")
include("./RPCA_Solver.jl")

# generate data
m,n,k = 500,100,5
Ut    = rand(k,m)
Vt    = rand(k,n)
A     = Ut.'*Vt
κt    = 0.08
for I in eachindex(A)
    A[I] += sampleH(κt)
end

# initialization
U = zeros(k,m)
V = ones(k,n)
# U = copy(Ut)
# V = copy(Vt)
# U = rand(k,m)
# V = rand(k,n)
κ = 0.5
λ = 0.0

params = solver_params(1e-8,100,10)
RPCA_Solver!(U,V,κ,A,λ,params)
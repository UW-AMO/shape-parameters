#======= Running Script for Quantile =======#

# import module and function
include("./utility.jl")
push!(LOAD_PATH,"../../src")
using PLQShape

# generate data and construct PLQShape objective
srand(123)
m,n,τ = 200,10,0.9
A,xt,y = genData(m,n,τ)

# PLQ primal variable
x = zeros(n)
θ = [0.5]

function ρ(x,θ,A,y)
    τ = θ[1];
    r = A*x - y;
    m = length(y);
    val = 0.0;
    for i = 1:m
        val += ifelse(r[i] > 0.0, r[i]*(1-τ), -r[i]*τ)
    end
    return val
end

function lognc(θ)
    τ = θ[1];
    return -log(τ*(1-τ))
end

function L₁(θ,A)
    return Inf
end

S = [-1.0 1.0]
s = [0.0;1.0]


primal = PLQ_Primal(-A,-y,x,θ,S,s,ρ,L₁,lognc)

# PLQ dual variable
B = eye(m)
G = zeros(1,m)
b = zeros(m)
M = zeros(m,m)

C = [-eye(m) eye(m)]
H = [ones(1,m) -ones(1,m)]
c = [zeros(m);ones(m)]

dual = PLQ_Dual(B,G,b,M,C,H,c)

# IPsolve solver
params = IP_params(0.1,0.1,1e-6,20)
x,θ = IPsolve(primal,dual,params)

# test result
@show θ
@show vecnorm(x-xt)/vecnorm(xt)

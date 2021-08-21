#= ==== Generate the Table in the Paper ==== =#

###### import module and function #####
include("./utility.jl")
push!(LOAD_PATH,"../../src")
using PLQShape
using Convex
using SCS
using COSMO
using Random
using LinearAlgebra

###### generate data and construct PLQShape objective #####
Random.seed!(777)
m, n = 1000, 50
κ = 1.0
# τ = [0.1,0.2,0.5,0.8,0.9]
τ = [0.5]
num_trials = 1

# use IPsolve to obtain result

# primal variables
S = -Matrix{Float64}(I, 2, 2);
s = [0.0;0.0];

function ρ(r, θ)
    τ = θ[1];
    κ = θ[2];
    m = length(r);
    val = 0.0;
    for i = 1:m
        r[i] <    -τ * κ ? val += -τ * κ * r[i] - 0.5 * τ^2 * κ^2 :
        r[i] > (1 - τ) * κ ? val += (1 - τ) * κ * r[i] - 0.5 * (1 - τ)^2 * κ^2 :
                         val += 0.5 * r[i]^2
    end
    return val
end

function L₁(θ, A)
    return norm(A, 2)^2
end

function lognc(θ)
    a = θ[1];
    b = θ[2];
    val = log(exp(-0.5 * a^2) / a +
        sqrt(0.5 * pi) * (erf(a / sqrt(2)) + erf(b / sqrt(2))) +
        exp(-0.5 * b^2) / b);
    return val
end

# dual variables
B = Matrix{Float64}(I, m, m)
G = zeros(2, m)
b = zeros(m)
M = Matrix{Float64}(I, m, m)

C = [-Matrix{Float64}(I, m, m) Matrix{Float64}(I, m, m)]
H = [ones(1, m) zeros(1, m);zeros(1, m) ones(1, m)]
c = zeros(2 * m)

# solver parameters
dual   = PLQ_Dual(B, G, b, M, C, H, c)
params = IP_params(1.0, 0.1, 1e-6, 50, zeros(51))

# try different τ
errs_qh = zeros(length(τ), num_trials)
errs_ls = zeros(length(τ), num_trials)
errs_lad = zeros(length(τ), num_trials)
for j = 1:num_trials
    for i = 1:length(τ)
        @show(τ[i],κ)
        A, xt, y = genData(m, n, [τ[i],κ])
        primal = PLQ_Primal(-A, -y, xt, [0.5,0.5], S, s, ρ, L₁, lognc)
        params.convergent_history = zeros(51)
        x, θ = IPsolve(primal, dual, params)
        α, β = θ[1], θ[2]
        θ = [α / (α + β),α + β]
        @show θ
        @show norm(x - xt) / norm(xt)
        xl = A \ y
        @show norm(xl - xt) / norm(xt)
        x_lad = Variable(length(xt))
        cost = norm(y - A * x_lad, 1)
        prob = minimize(cost)
        solve!(prob, SCS.Optimizer)
        x_lad = x_lad.value[:]
        errs_ls[i, j] = norm(xl - xt) / norm(xt)
        errs_qh[i, j] = norm(x - xt) / norm(xt)
        errs_lad[i, j] = norm(x_lad - xt) / norm(xt)
    end
end
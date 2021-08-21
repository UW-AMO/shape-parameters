#= 
    Simulation for consistency =#
include("./utility.jl")
push!(LOAD_PATH,"../../src")
using PLQShape
using Convex
using SCS
using Random
using LinearAlgebra

# set random seed
Random.seed!(777)

# simulate data
n = 50
num_trials = 10
m_array = collect(500:500:5000)
# m_array = [1000]
xt = randn(n)
θt = [0.1, 1.0]


θ_qh = zeros(2, length(m_array), num_trials)
x_qh = zeros(length(xt), length(m_array), num_trials)
x_ls = zeros(length(xt), length(m_array), num_trials)
x_lad = zeros(length(xt), length(m_array), num_trials)
err_qh = zeros(length(m_array), num_trials)
err_ls = zeros(length(m_array), num_trials)
err_lad = zeros(length(m_array), num_trials)
for j = 1:num_trials
    for i = 1:length(m_array)
        m = m_array[i]
        A = randn(m, n)
        y = A * xt
        for i = 1:m
            # y[i] += randn()
            y[i] += sampleQH(θt)
        end

        # least square
        x_ls[:, i, j] = A \ y
        err_ls[i, j] = norm(x_ls[:, i, j] - xt) / norm(xt)

        # least absolute value
        x = Variable(length(xt))
        x.value = copy(xt)
        
        cost = norm(y - A * x, 1)
        prob = minimize(cost)
        solve!(prob, SCS.Optimizer, warmstart=true)
        x_lad[:, i, j] = x.value[:]
        err_lad[i, j] = norm(x_lad[:, i, j] - xt) / norm(xt)

        # IPsolve
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
                r[i] < -τ * κ ? val += -τ * κ * r[i] - 0.5 * τ^2 * κ^2 :
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

        primal = PLQ_Primal(-A, -y, xt, [0.5, 0.5], S, s, ρ, L₁, lognc)
        params.convergent_history = zeros(51)
        x_qh[:, i, j], θ = IPsolve(primal, dual, params)
        α, β = θ[1], θ[2]
        θ_qh[:, i, j] = [α / (α + β),α + β]
        err_qh[i, j] = norm(x_qh[:, i, j] - xt) / norm(xt)
    end
end
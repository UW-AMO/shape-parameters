#= ===================================================================
    Solve RPCA Objective
    min_{L,R,κ}
    ρ(UᵀV - A; [κ,σ])/2 + λ(‖U‖_F² + ‖V‖_F²) + m⋅n⋅log[c([κ,σ])]
=================================================================== =#
# include("../../../1Dmin/rminbnd.jl")
using ForwardDiff
using NLopt
##### data type #####
mutable struct solver_params
    tol::Float64
    itermax::Int64
    pf::Int64
end

##### main solver #####
function RPCA_Solver!(U, V, θ, A, λ, params;fixθ=false)
    # problem dimension
    m, n = size(A)
    k   = size(U, 1)
    # N   = 0.1*randn(m,n)
    # pre-allocate variables
    R   = U' * V - A
    u   = Array(SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true}, m)
    v   = Array(SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true}, n)
    for i = 1:m u[i] = view(U, :, i) end
    for j = 1:n v[j] = view(V, :, j) end

    gu  = zeros(k)
    gv  = zeros(k)
    obj = objective(R, U, V, θ, λ)
    err = 1.0
    noi = 0
    while err ≥ params.tol
        ##### update U #####
        η = 1.0 / (norm(V)^2 / θ[2]^2 + λ)     # step size
        for i = 1:m
            # update gradient
            copy!(gu, u[i]); scale!(gu, λ);
            for j = 1:n
                r = dot(u[i], v[j]) - A[i,j]
                r = ∇ρ(r, θ)
                BLAS.axpy!(r, v[j], gu)
            end
            # update uᵢ
            BLAS.axpy!(-η, gu, u[i])
            # projected onto [0,1]
            for ii = 1:k
                u[i][ii] < 0.0 ? u[i][ii] = 0.0 :
                u[i][ii] > 1.0 ? u[i][ii] = 1.0 : continue
            end
        end
        # copy!(R,A)
        # BLAS.gemm!('T','N',1.0,U,V,-1.0,R)
        # @show objective(R,U,V,θ,λ)
        ##### update V #####
        η = 1.0 / (norm(U)^2 / θ[2]^2 + λ)     # step size
        for j = 1:n
            # update gradient
            copy!(gv, v[j]); scale!(gv, λ);
            for i = 1:m
                r = dot(u[i], v[j]) - A[i,j]
                r = ∇ρ(r, θ)
                BLAS.axpy!(r, u[i], gv)
            end
            # update vⱼ
            BLAS.axpy!(-η, gv, v[j])
            # projected onto [0,1]
            for jj = 1:k
                v[j][jj] < 0.0 ? v[j][jj] = 0.0 :
                v[j][jj] > 1.0 ? v[j][jj] = 1.0 : continue
            end
        end
        # copy!(R,A)
        # BLAS.gemm!('T','N',1.0,U,V,-1.0,R)
        # @show objective(R,U,V,θ,λ)
        ##### update κ #####
        copy!(R, A)
        BLAS.gemm!('T', 'N', 1.0, U, V, -1.0, R)
        fixθ || (θ = projθ(R, θ))
        ##### update convergent history #####
        noi += 1
        obn = objective(R, U, V, θ, λ)
        err = obj - obn
        obj = obn
        if noi % params.pf == 0
            @printf("iter %3d, obj %1.5e, κ %1.2e, σ %1.2e, err %1.5e\n",
            noi, obj, θ[1], θ[2], err)
        end
        if noi ≥ params.itermax
            println("Reach Maximum Iteration!")
            break
        end
    end
    return θ
end

##### support function #####
function θfun(R, θ)
    m, n = size(R);
    κ, σ = θ[1], θ[2];
    val = 0.0;
    for I in eachindex(R)
        val += ifelse(abs(R[I]) > κ * σ, κ * abs(R[I]) / σ - 0.5 * κ^2, 0.5 * R[I]^2 / σ^2)
    end
    c   = 2.0 * exp(-0.5 * κ^2) / κ + sqrt(2.0 * pi) * erf(κ / sqrt(2));
    val += m * n * (log(c) + log(abs(σ)))
    return val
end

function ∇ρ(r, θ)
    κ = θ[1]
    σ = θ[2]
    r < -κ * σ ? (return -κ / σ) :
    r >  κ * σ ? (return  κ / σ) : (return r / σ^2)
end

function projθ(R, θ)
    # count = 0
    function θfun(θ::Vector, g::Vector, R::Matrix)
        n   = length(R);
        κ, σ = θ[1], θ[2];
        val = 0.0;
        # gradient and hessian of huber penalty w.r.t. κ
        for I in eachindex(R)
            if abs(R[I]) > κ * σ
                # gradient
                g[1] += abs(R[I]) / σ - κ;
                g[2] -= κ * abs(R[I]) / σ^2;
                val  += κ * abs(R[I]) / σ - 0.5 * κ^2;
            else
                # gradient
                g[2] -= R[I]^2 / σ^3;
                val  += 0.5 * R[I]^2 / σ^2;
            end
        end
        val /= n;
        scale!(g, 1 / n);
        # gradient and hessian of log normalization constant
        eκ  = exp(-0.5 * κ^2);
        c   = 2.0 * eκ / κ + sqrt(2.0 * pi) * erf(κ / sqrt(2.0));
        dc  = -2.0 * eκ / κ^2;
        ddc = (4.0 / κ^3 + 2.0 / κ) * eκ;
        g[1] += dc / c;
        g[2] += 1.0 / σ;
        val  += log(c) + log(σ);

        # global count
        # count::Int += 1
        # println("f($θ) = $val")

        return val
    end

    myfunc = (θ::Vector, g::Vector) -> θfun(θ, g, R)

##### Call Solver #####
    opt = Opt(:LD_MMA, 2)
    lower_bounds!(opt, [1e-4,1e-4])
    xtol_rel!(opt, 1e-6)

    min_objective!(opt, myfunc)
    (minf, minx, ret) = optimize(opt, [1e-2,1e-2])
    return minx
end

function gradHθ(R, θ)
    m, n = size(R)
    κ, σ = θ[1], θ[2]
    # gradient and hessian of huber penalty w.r.t. κ
    g   = zeros(2)
    H   = zeros(2, 2)
    for I in eachindex(R)
        if abs(R[I]) > κ * σ
            # gradient
            g[1] += abs(R[I]) / σ - κ
            g[2] -= κ * abs(R[I]) / σ^2
            # hessian
            H[1,1] -= 1.0
            H[1,2] -= abs(R[I]) / σ^2
            H[2,1] -= abs(R[I]) / σ^2
            H[2,2] += 2.0 * κ * abs(R[I]) / σ^3
        else
            # gradient
            g[2] -= R[I]^2 / σ^3
            # hessian
            H[2,2] += 3 * R[I]^2 / σ^4
        end
    end
    scale!(g, 1 / (m * n))
    scale!(H, 1 / (m * n))
    # gradient and hessian of log normalization constant
    eκ  = exp(-0.5 * κ^2)
    c   = 2.0 * eκ / κ + sqrt(2.0 * pi) * erf(κ / sqrt(2.0))
    dc  = -2.0 * eκ / κ^2
    ddc = (4.0 / κ^3 + 2.0 / κ) * eκ
    g[1] += dc / c
    g[2] += 1.0 / σ
    H[1,1] += (ddc * c - dc^2) / c^2
    H[2,2] -= 1.0 / σ^2

    return g, H
end

function objective(R, U, V, θ, λ)
    κ   = θ[1]
    σ   = θ[2]
    c   = σ * (2.0 * exp(-0.5 * κ^2) / κ + sqrt(2.0 * pi) * erf(κ / sqrt(2)));
    m, n = size(R)
    val = 0.0
    for I in eachindex(R)
        val += ifelse(abs(R[I]) > κ * σ, κ * abs(R[I]) / σ - 0.5 * κ^2, 0.5 * R[I]^2 / σ^2)
    end
    nU  = 0.0
    for I in eachindex(U)
        nU += U[I]^2
    end
    val += 0.5 * λ * nU
    nV  = 0.0
    for I in eachindex(V)
        nV += V[I]^2
    end
    val += 0.5 * λ * nV
    val /= (m * n)
    val += log(c)
    return val
end
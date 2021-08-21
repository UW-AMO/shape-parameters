#= ===================================================================
    Solve RPCA Objective
    min_{L,R,κ} ρ(UᵀV - A; κ)/2 + λ(‖U‖_F² + ‖V‖_F²) + m⋅n⋅log[c(κ)]
=================================================================== =#
# include("../../../1Dmin/rminbnd.jl")
using Optim
##### data type #####
mutable struct solver_params
    tol::Float64
    itermax::Int64
    pf::Int64
end

##### main solver #####
function RPCA_Solver!(U, V, κ, A, λ, params;fixκ=false)
    # problem dimension
    m, n = size(A)
    k   = size(U, 1)

    # pre-allocate variables
    R   = U' * V - A
    u   = Array(SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true}, m)
    v   = Array(SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true}, n)
    for i = 1:m u[i] = view(U, :, i) end
    for j = 1:n v[j] = view(V, :, j) end

    gu  = zeros(k)
    gv  = zeros(k)
    obj = objective(R, U, V, κ, λ)
    err = 1.0
    noi = 0
    while err ≥ params.tol
        ##### update U #####
        η = 1.0 / (norm(V)^2 + λ)     # step size
        for i = 1:m
            # update gradient
            copy!(gu, u[i]); scale!(gu, λ);
            for j = 1:n
                r = dot(u[i], v[j]) - A[i,j]
                r = ∇ρ(r, κ)
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

        ##### update V #####
        η = 1.0 / (norm(U)^2 + λ)     # step size
        for j = 1:n
            # update gradient
            copy!(gv, v[j]); scale!(gv, λ);
            for i = 1:m
                r = dot(u[i], v[j]) - A[i,j]
                r = ∇ρ(r, κ)
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

        ##### update κ #####
        copy!(R, A)
        BLAS.gemm!('T', 'N', 1.0, U, V, -1.0, R)
        fixκ || (κ = projκ(R, κ))
        ##### update convergent history #####
        noi += 1
        obn = objective(R, U, V, κ, λ)
        err = obj - obn
        obj = obn
        if noi % params.pf == 0
            @printf("iter %3d, obj %1.5e, κ %1.2e, err %1.5e\n", noi, obj, κ, err)
        end
        if noi ≥ params.itermax
            println("Reach Maximum Iteration!")
            break
        end
    end
    return κ
end

##### support function #####
function κfun(R, κ)
    m, n = size(R);
    val = 0.0;
    for I in eachindex(R)
        val += ifelse(abs(R[I]) > κ, κ * abs(R[I]) - 0.5 * κ^2, 0.5 * R[I]^2);
    end
    val /= m * n;
    c   = 2.0 * exp(-0.5 * κ^2) / κ + sqrt(2.0 * pi) * erf(κ / sqrt(2));
    # α   = 2.335*1e-5;
    val += log(c);
    return val
end

function ∇ρ(r, κ)
    r < -κ ? (return -κ) :
    r >  κ ? (return  κ) : (return r)
end

function projκ(R, κ)
    # newton solver to project κ
    # itermax = 50
    # tol = 1e-6
    # g,H = gradHκ(R,κ)
    # err = abs(g)
    # noi = 0
    # # @show err
    # while err ≥ tol
    #     κ = κ - g/H
    #     κ < 0.0 && (κ = 1.0/(noi + 1.0))
    #     g,H = gradHκ(R,κ)
    #     err = abs(g)
    #     noi += 1
    #     if noi ≥ itermax
    #         println("projκ Reach Maximum Iteration!")
#         break
    #     end
    # end
    # f = k -> κfun(R,k)
    # κ,fval = rminbnd(f,0.0,1.0,κ)
    res = optimize(k -> κfun(R, k), 0.0, 1.0);
    return res.minimizer
    # return 0.1
            end

function gradHκ(R, κ)
    m, n = size(R)
    # gradient and hessian of huber penalty w.r.t. κ
    g   = 0.0
    H   = 0.0
    for I in eachindex(R)
        if abs(R[I]) > κ
            g += abs(R[I]) - κ
            H -= 1.0
        end
    end
    g /= m * n;
    H /= m * n;
    # gradient and hessian of log normalization constant
    eκ  = exp(-0.5 * κ^2)
    c   = 2.0 * eκ / κ + sqrt(2.0 * pi) * erf(κ / sqrt(2.0))
    dc  = -2.0 * eκ / κ^2
    ddc = (4.0 / κ^3 + 2.0 / κ) * eκ
    g  += dc / c
    H  += (ddc * c - dc^2) / c^2

    return g, H
end
        
function objective(R, U, V, κ, λ)
    c   = 2.0 * exp(-0.5 * κ^2) / κ + sqrt(2.0 * pi) * erf(κ / sqrt(2));
    m, n = size(R)
    nU  = 0.0
    val = 0.0
    for I in eachindex(R)
        val += ifelse(abs(R[I]) > κ, κ * abs(R[I]) - 0.5 * κ^2, 0.5 * R[I]^2)
    end
    val /= m * n;
    # for I in eachindex(U)
    #     nU += U[I]^2
    # end
    # val += λ*nU
    # nV  = 0.0
    # for I in eachindex(V)
    #     nV += V[I]^2
    # end
    # val += λ*nV
    val += log(c)
    return val
end
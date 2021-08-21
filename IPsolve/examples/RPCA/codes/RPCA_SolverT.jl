#= ===================================================================
    Solve RPCA Objective
    min_{L,R,κ}
    ρ(UᵀV - A; [κ,σ])/2 + λ(‖U‖_F² + ‖V‖_F²) + m⋅n⋅log[c([κ,σ])]
=================================================================== =#
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
        η = 1.0 / (2 * norm(V)^2 / θ[2]^2 + λ)     # step size
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
        η = 1.0 / (2 * norm(U)^2 / θ[2]^2 + λ)     # step size
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
    κ, σ = θ[1], θ[2]
    n = length(R)
    val = 0.0
    for I in eachindex(R)
        abs(R[I]) > κ * σ ? val += 2.0 * κ / (σ * (1.0 + κ^2)) * (abs(R[I]) - κ * σ) + log(1.0 + κ^2) :
                          val += log(1.0 + R[I]^2 / σ^2)
    end
    val /= n
    val += log(σ * (1 + 2 * κ * atan(κ)) / κ)
    return val
end

function ∇ρ(r, θ)
    κ = θ[1]
    σ = θ[2]
    r < -κ * σ ? (return -2 * κ / (σ * (1 + κ^2))) :
    r >  κ * σ ? (return  2 * κ / (σ * (1 + κ^2))) : (return 2 * r / (σ^2 + r^2))
end

function projθ(R, θ)
    # function θfun(θ::Vector, g::Vector, R::Matrix)
    #     n   = length(R);
    #     κ,σ = θ[1],θ[2];
    #     val = 0.0;
    #     # gradient and hessian of huber penalty w.r.t. κ
    #     for I in eachindex(R)
    #         if abs(R[I]) > κ*σ
    #             # gradient
    #             g[1] += 2*(κ^2-1)*(κ*σ-abs(R[I]))/((1+κ^2)^2*σ)
    #             g[2] -= 2*κ*abs(R[I])/((1+κ^2)*σ^2)
    #             val  += 2.0*κ/(σ*(1.0+κ^2))*(abs(R[I])-κ*σ) + log(1.0+κ^2);
    #         else
    #             # gradient
    #             g[2] -= 2*R[I]^2/(R[I]^2*σ+σ^3)
    #             val  += log(1.0 + R[I]^2/σ^2)
    #         end
    #     end
    #     val /= n;
    #     scale!(g,1/n);
    #     # gradient and hessian of log normalization constant
    #     g[1] += (κ^2-1)/(κ*(1+κ^2)*(1+2*κ*atan(κ)))
    #     g[2] += 1/σ
    #     val  += log(σ*(1+2*κ*atan(κ))/κ);

    #     # global count
    #     # count::Int += 1
    #     # println("f($θ) = $val")

    #     return val
    # end

    # myfunc = (θ::Vector,g::Vector) -> θfun(θ,g,R)

    # ##### Call Solver #####
    # opt = Opt(:LD_MMA, 2)
    # lower_bounds!(opt,[1e-4,1e-4])
    # xtol_rel!(opt,1e-6)

    # min_objective!(opt,myfunc)
    # (minf,minx,ret) = optimize(opt,θ)
    # return minx
    tol = 1e-6;
    itm = 50;
    θ   = [5.5,0.025];
    g, H = gradHθ(θ, R);
    err = vecnorm(g);
    noi = 0;
    while err ≥ tol
        # update direction
        noi ≥ 10 ? p = H \ g : p = g / norm(H);
        # p = H \ g
        # line search
        η = 1.0;
        p[1] > θ[1] && (η = min(η, θ[1] / p[1] * 0.99));
        p[2] > θ[2] && (η = min(η, θ[2] / p[2] * 0.99));
    # update θ
        θ = θ - η * p;
        # update history
        g, H = gradHθ(θ, R);
        # obj = θfun(R,θ);
        err = vecnorm(g);
        noi = noi + 1;
        # noi%1==0 && @printf("iter %3d, obj %1.5e, err %1.5e\n",noi,obj,err)
        if noi ≥ itm
            break;
        end
    end
    return θ
end

function gradHθ(θ, R)
    n   = length(R)
    κ, σ = θ[1], θ[2]
    g   = zeros(2)
    H   = zeros(2, 2)
    # gradient and hessian of the penalty
    for i in eachindex(R)
        if abs(R[i]) > κ * σ
            # gradient
            g[1] += 2 * (κ^2 - 1) * (κ * σ - abs(R[i])) / ((1 + κ^2)^2 * σ)
            g[2] -= 2 * κ * abs(R[i]) / ((1 + κ^2) * σ^2)
            # hessian
            H[1,1] -= 2 * (6 * κ * abs(R[i]) - 2 * κ^3 * abs(R[i]) + σ - 6 * κ^2 * σ + κ^4 * σ) / ((1 + κ^2)^3 * σ)
            H[1,2] += 2 * (κ^2 - 1) * abs(R[i]) / ((1 + κ^2)^2 * σ^2)
            H[2,1] += 2 * (κ^2 - 1) * abs(R[i]) / ((1 + κ^2)^2 * σ^2)
            H[2,2] += 4 * κ * abs(R[i]) / ((1 + κ^2) * σ^3)
        else
            # gradient
            g[2] -= 2 * R[i]^2 / (R[i]^2 * σ + σ^3)
            # hessian
            H[2,2] += 2 * (R[i]^4 + 3 * R[i]^2 * σ^2) / (σ^2 * (R[i]^2 + σ^2)^2)
        end
    end
    scale!(g, 1 / n)
    scale!(H, 1 / n)
    # gradient and hessian of normalization constant
    # gradient
    g[1] += (κ^2 - 1) / (κ * (1 + κ^2) * (1 + 2 * κ * atan(κ)))
    g[2] += 1 / σ
    # hessian
    H[1,1] += (1 + 6 * κ^2 - 3 * κ^4 + (4 * κ + 8 * κ^3 - 4 * κ^5) * atan(κ)) / (κ^2 * (1 + κ^2)^2 * (1 + 2 * κ * atan(κ))^2)
    H[2,2] -= 1 / σ^2
    
    return g, H
end

function objective(R, U, V, θ, λ)
    κ   = θ[1]
    σ   = θ[2]
    val = θfun(R, θ)
    # disable regularizer for now
    # nU  = 0.0
    # for I in eachindex(U)
    #     nU += U[I]^2
    # end
    # val += 0.5*λ*nU
    # nV  = 0.0
    # for I in eachindex(V)
    #     nV += V[I]^2
    # end
    # val += 0.5*λ*nV
    return val
end
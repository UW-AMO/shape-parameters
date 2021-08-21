#====================================================================
    PALM Method with Projection

    Solve problem,
    (𝒫) min_{x,θ ∈ 𝒟} ρ(Ax - y;θ) + m⋅log[c(θ)]
====================================================================#
using ForwardDiff
using Printf

##### PALMProj Parameters #####
mutable struct PALMProj_params
    tol::Float64
    iter_max::Int64
    nt_tol::Float64
    nt_iter_max::Int64
    convergent_history::Array{Float64,1}
end

##### PALM with Projection Solver #####
function PALMProj(primal::PLQ_Primal, params::PALMProj_params)

    # primal variables
    A = primal.A; y = primal.y;
    x = primal.x; θ = primal.θ;
    m,n = size(A)

    # PALM projection parameters
    tol = params.tol;
    iter_max = params.iter_max;
    nt_tol = params.nt_tol;
    nt_iter_max = params.nt_iter_max;
    convergent_history = params.convergent_history

    # PALM iterations
    noi = 0;
    err = 1.0;
    r   = y - A*x;
    convergent_history[1] = primal.ρ(r,θ) + m*primal.lognc(θ)
    while err ≥ tol
        # calculate Lipschitz constant
        c  = primal.L₁(θ,A)
        # update x
        f(r) = primal.ρ(r,θ)
        g  = transpose(A)*ForwardDiff.gradient(f,r) # gradient
        xn = x + g/c                    # gradient descent step
        r  = y - A*xn                       # update residual
        # update θ
        # θn = copy(θ)
        θn = projθ(r,θ,primal, nt_tol, nt_iter_max)
        # update convergent history
        noi += 1
        # err = vecnorm(θn-θ) + vecnorm(xn-x)
        err = vecnorm(g)
        copy!(x,xn)
        copy!(θ,θn)
        obj = primal.ρ(r,θ) + m*primal.lognc(θ)
        @printf("iter %3d, obj %1.5e, err %1.5e\n",noi,obj,err)
        convergent_history[noi+1] = obj
        if noi ≥ iter_max
            println("Reach Maximum Iterations!")
            break
        end
    end
    params.convergent_history = convergent_history[1:noi+1]
end

##### Project Fuction #####
function projθ(r, θ, primal, nt_tol, nt_iter_max)
    m,n = size(primal.A)
    f(θ) = primal.ρ(r,θ)
    S = primal.S
    s = primal.s
    μ = 1.0
    # newton solver for interior point method
    H  = ForwardDiff.hessian(f,θ)  + m*ForwardDiff.hessian(primal.lognc,θ)
    g  = ForwardDiff.gradient(f,θ) + m*ForwardDiff.gradient(primal.lognc,θ)
    # KKT system
    d   = s - S'*θ; k = length(d); q   = ones(k);
    Fμ  = [diagm(d)*q - μ*ones(k); g + S*q]
    ∇Fμ = [diagm(d) -diagm(q)*S'; S H]
    idq = 1:k; idθ = k+1:k+length(θ);
    z   = [q; θ]
    # start iteration
    nt_noi = 0
    nt_err = vecnorm(g)
    while nt_err ≥ nt_tol
        # calculate direction
        p  = ∇Fμ \ Fμ
        # line search
        η  = 1.0
        dd = -S'*p[idθ]
        for i = 1:k
            dd[i] ≤ 0.0 && continue
            η = min(η,d[i]/dd[i])
        end
        η -= 1e-2*μ
        z -= η*p
        # update variable gradient hessian
        θ  = z[idθ]; q = z[idq];
        H  = ForwardDiff.hessian(f,θ)  + m*ForwardDiff.hessian(primal.lognc,θ)
        g  = ForwardDiff.gradient(f,θ) + m*ForwardDiff.gradient(primal.lognc,θ)
        d  = s - S'*θ
        Fμ  = [diagm(d)*q - μ*ones(k); g + S*q]
        ∇Fμ = [diagm(d) -diagm(q)*S'; S H]
        # update convergence information
        nt_noi += 1
        nt_err = vecnorm(Fμ)
        μ  = 0.1*sum(d.*q)/k
        if nt_noi ≥ nt_iter_max
            println("Newton Method Reach Maximum Iterations!")
            break
        end
        # obj = f(θ) + m*primal.lognc(θ)
        # @printf("iter %2d, obj %1.5e, err %1.5e\n", nt_noi, obj, nt_err)
    end
    return θ
end
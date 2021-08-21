#====================================================================
    PALM Method

    Solve problem,
    (𝒫) min_{x,θ ∈ 𝒟} ρ(Ax - y;θ)/m + log[c(θ)]
    with
        H(x,θ) = ρ(Ax - y;θ)
        h(θ)   = δ_𝒟(θ) + log[c(θ)]
    no regularizer for x for now.
====================================================================#
using ForwardDiff
using Optim
using Printf

##### PALM Parameters #####
mutable struct PALM_params
    tol::Float64
    itm::Int64
    nt_tol::Float64
    nt_itm::Int64
    convergent_history::Array{Float64,1}
end

##### PALM Solver #####
function PALM(primal::PLQ_Primal, params::PALM_params)

    # primal variables
    A = primal.A; y = primal.y;
    x = primal.x; θ = primal.θ;
    m,n = size(A)

    # PALM parameters
    tol = params.tol;
    itm = params.itm;
    nt_tol = params.nt_tol;
    nt_itm = params.nt_itm;
    convergent_history = params.convergent_history;

    # PALM iterations
    noi = 0
    err = 1.0;
    r   = y - A*x;
    convergent_history[1] = primal.ρ(r,θ) + m*primal.lognc(θ);
    while err ≥ tol
        # calculate Lipschitz constant
        c  = primal.L₁(θ,A);
        d  = 1.0;
        # update x
        f(r) = primal.ρ(r,θ);
        gx = transpose(A)*ForwardDiff.gradient(f,r); # gradient
        xn = x + gx/c;                      # gradient descent step
        r  = y - A*xn;                      # update residual
        # update θ
        h(θ) = primal.ρ(r,θ)
        gθ = ForwardDiff.gradient(h,θ);
        θn = θ - gθ/d;
        θn = prox_lognc(θn, primal.lognc, d, m, nt_tol, nt_itm);
        err = vecnorm(θn-θ) + vecnorm(xn-x)
        # err = vecnorm(g)
        noi += 1;
        copy!(x,xn)
        copy!(θ,θn)
        obj = primal.ρ(r,θ) + m*primal.lognc(θ)
        @printf("iter %3d, obj %1.5e, err %1.5e\n",noi,obj,err)
        convergent_history[noi+1] = obj
        if noi ≥ itm
            println("Reach Maximum Iterations!")
            break
        end
    end
    params.convergent_history = convergent_history[1:noi+1]
    return θ
end

##### proximal function #####
function prox_lognc(θn, lognc, d, m, nt_tol, nt_itm)
    f = θ -> 0.5*vecnorm(θ - θn)^2 + m*lognc(θ)/d;
    # res = optimize(f,[1.0,1.0],BFGS());
    # return res.minimizer
    θ = [1.0,1.0];
    g = ForwardDiff.gradient(f,θ);
    H = ForwardDiff.hessian(f,θ);
    err = vecnorm(g);
    noi = 0;
    while err ≥ nt_tol
        p = H \ g;
        η = 1.0;
        p[1] > θ[1] && (η = min(η,θ[1]/p[1]*0.99));
        p[2] > θ[2] && (η = min(η,θ[2]/p[2]*0.99));
        # update θ
        θ = θ - η*p;
        g = ForwardDiff.gradient(f,θ);
        H = ForwardDiff.hessian(f,θ);
        # update convergent history
        noi+= 1;
        obj = f(θ);
        err = vecnorm(g);
        # @printf("iter %2d, obj %1.5e, err %1.5e\n", noi, obj, err);
        if noi ≥ nt_itm
            println("Maximum Iteration!")
            break;
        end
    end
    return θ
end
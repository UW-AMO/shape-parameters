#= ===================================================================
    Interior Point Method

    Solve problem,
    (𝒫) min_{x,Sᵀθ ≤ s}
        sup_{Cᵀu ≤ Hᵀθ + c}{uᵀ[B(A⋅x-y) - Gᵀθ - b] - uᵀ⋅M⋅u/2}
        + m⋅log[c(θ)]
=================================================================== =#
using ForwardDiff
using Printf
using LinearAlgebra

##### IPsolve parameters #####
mutable struct IP_params
    μ::Float64          # coefficient of barrier function
    α::Float64          # rate of μ ⟶ 0⁺
    tol::Float64        # tolerance
    iter_max::Int64     # maximum number of iterations
    convergent_history::Array{Float64,1}
end

##### IPsolver #####
function IPsolve(primal::PLQ_Primal, dual::PLQ_Dual, params::IP_params)

    # primal variables
    A = primal.A; y = primal.y;
    x = primal.x; θ = primal.θ;
    S = primal.S; s = primal.s;

    # dual variables
    B = dual.B; M = dual.M;
    c = dual.c; b = dual.b;
    G = dual.G; C = dual.C;
    H = dual.H;
    
    # interior point parameters
    μ = params.μ;
    α = params.α;
    tol = params.tol;
    iter_max = params.iter_max;
    convergent_history = params.convergent_history;

    # allocate other variables
    # dimensions
    m, n = size(A);
    nq = size(C, 2) + size(S, 2);
    nu = size(M, 1);
    nx = n;
    nθ = length(θ);
    # KKT system
    z = [ones(nq);zeros(nu);x;θ];
    idq = 1:nq;
    idu = nq + 1:nq + nu;
    idx = nq + nu + 1:nq + nu + nx;
    idθ = nq + nu + nx + 1:nq + nu + nx + nθ;
    Fμ = zeros(nq + nu + nx + nθ);
    ∇Fμ = zeros(nq + nu + nx + nθ, nq + nu + nx + nθ);

    # initialize the KKT system
    # variables that don't change
    Fuq = view(∇Fμ, idu, idq); copy!(Fuq, [-C zeros(nu, size(S, 2))]);
    Fuu = view(∇Fμ, idu, idu); copy!(Fuu, -M);
    Fux = view(∇Fμ, idu, idx); copy!(Fux, B * A);
    Fuθ = view(∇Fμ, idu, idθ); copy!(Fuθ, -G');
    Fxu = view(∇Fμ, idx, idu); copy!(Fxu, transpose(Fux));
    Fθq = view(∇Fμ, idθ, idq); copy!(Fθq, [H S]);
    Fθu = view(∇Fμ, idθ, idu); copy!(Fθu, -G);
    # update the rest Fμ, ∇Fμ
    updateKKT!(Fμ, ∇Fμ, z, μ, α, primal, dual, idq, idu, idx, idθ, update_μ=false);

    # interior point iterations
    noi = 0;
    err = norm(Fμ);
    r = A * z[idx] - y;
    convergent_history[1] = primal.ρ(r, z[idθ]) + m * primal.lognc(z[idθ])
    while err ≥ tol
        # 1 newton step
        p = ∇Fμ \ Fμ;
        # postive line search
        η = pos_ls(z, p, μ, primal, dual, idu, idθ, idq);
        # move one step forward
        BLAS.axpy!(-η, p, z)
        # update convergent information
        noi += 1
        μ = updateKKT!(Fμ, ∇Fμ, z, μ, α, primal, dual, idq, idu, idx, idθ);
        err = norm(Fμ);
        r = A * z[idx] - y;
        obj = primal.ρ(r, z[idθ]) + m * primal.lognc(z[idθ])
        @printf("iter %2d, obj, %1.5e, err %1.5e, μ %1.5e\n",noi,obj,err,μ)
        convergent_history[noi + 1] = obj
        if noi ≥ iter_max
            println("Reach Maximum Iterations!")
            break
        end
    end
    params.convergent_history = convergent_history[1:noi + 1]
    return z[idx], z[idθ]
end

##### auxilary functions #####
# update Fμ and ∇Fμ
function updateKKT!(Fμ, ∇Fμ, z, μ, α, primal, dual, idq, idu, idx, idθ;update_μ=true)
    # rename parameter to shorten the expression
    # primal variables
    A = primal.A; y = primal.y;
    S = primal.S; s = primal.s;
    m, n = size(A)

    # dual variables
    B = dual.B; M = dual.M; c = dual.c;
    G = dual.G; C = dual.C;
    b = dual.b; H = dual.H;

    # KKT variables
    q = view(z, idq); Fq = view(Fμ, idq);
    u = view(z, idu); Fu = view(Fμ, idu);
    x = view(z, idx); Fx = view(Fμ, idx);
    θ = view(z, idθ); Fθ = view(Fμ, idθ);
    Fqq = view(∇Fμ, idq, idq);
    Fqu = view(∇Fμ, idq, idu); Fuq = view(∇Fμ, idu, idq);
    Fqθ = view(∇Fμ, idq, idθ); Fθq = view(∇Fμ, idθ, idq);
    Fθθ = view(∇Fμ, idθ, idθ);

    # slack variable
    d = [c - transpose(C) * u + transpose(H) * θ; s - transpose(S) * θ];
    update_μ && (μ = α * sum(d .* q) / length(d));
    # if μ < 0.0
    #     for i = 1:length(q)
    #         q[i] < 0.0 && println("q[i] is negative", q[i])
    #     end
    # end

    # update Fμ: Fq = D⋅q - μ⋅𝟙
    for i = idq Fq[i] = d[i] * q[i] - μ; end
    # update Fμ: Fu = B(A⋅x-y) - Gᵀθ - b - M⋅u + [-C 0]⋅q
    copy!(Fu, B * (A * x - y) - transpose(G) * θ - b - M * u + Fuq * q);
    # update Fμ: Fx = Aᵀ⋅Bᵀ⋅u
    copy!(Fx, transpose(A) * (transpose(B) * u));
    # update Fμ: Fθ = -G⋅u + m∇log[nc(θ)] + [H S]⋅q
    ∇lognc = ForwardDiff.gradient(primal.lognc, θ)
    copy!(Fθ, -G * u + m * ∇lognc + Fθq * q);

    # update ∇Fμ: Fqq = D
    copy!(Fqq, diagm(0 => d[:]));
    # update ∇Fμ: Fqu = Q*[-C 0]ᵀ
    copy!(Fqu, diagm(0 => q[:]) * transpose(Fuq));
    # update ∇Fμ: Fqθ = Q*[H S]ᵀ
    copy!(Fqθ, diagm(0 => q[:]) * transpose([H -S]));
    # update ∇Fμ: Fθθ = m⋅∇²log[nc(θ)]
    ∇²lognc = ForwardDiff.hessian(primal.lognc, θ)
    copy!(Fθθ, m * ∇²lognc);

    return μ
end

# positive line search
function pos_ls(z, p, μ, primal, dual, idu, idθ, idq)
    u = view(z, idu); du = view(p, idu);
    θ = view(z, idθ); dθ = view(p, idθ);
    q = view(z, idq); dq = view(p, idq);

    C = dual.C; H = dual.H; S = primal.S;
    c = dual.c; s = primal.s;
    d = [c - C' * u + H' * θ; s - S' * θ]; d = d[:];
    dd = [-C' * du + H' * dθ; -S' * dθ]; dd = dd[:];

    η = 1.0
    for i = 1:length(d)
        d[i] < 0.0 && println("d[i] < 0.0!")
        dd[i] ≤ 0.0 && continue
        η = min(η, d[i] / dd[i])
    end
    for i = 1:length(q)
        q[i] < 0.0 && println("q[i] < 0.0!")
        dq[i] ≤ 0.0 && continue
        η = min(η, q[i] / dq[i])
    end
    return 0.99 * η
end
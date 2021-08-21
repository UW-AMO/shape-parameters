#====================================================================
  Total Quantile Regression Solver
  using PALM algorithm
====================================================================#
#--------------------------------------------------------------------
# Objective
#--------------------------------------------------------------------
function obj(r, vaq, va, τ, m, n)
    val = 0.5*vecnorm(r)^2
    for j = 1:n
        for i = 1:m
        a = va[j][i] - vaq[j][i]
        a < 0.0 ? val -= τ[j]*a : val += (1-τ[j])*a
        end
        val -= m*(log(τ[j]*(1-τ[j])))
    end
    return val
end
#--------------------------------------------------------------------
# Project x
#--------------------------------------------------------------------
function projx!(x, AA, Aq, y)
    At_mul_B!(x,Aq,y)                   # A'y
    At_mul_B!(AA,Aq,Aq)                 # A'A
    LAPACK.sysv!('U',AA,x)              # (A'A)\(A'y)
end
#--------------------------------------------------------------------
# Update x
#--------------------------------------------------------------------
function updatex!(x, g, r, Aq)
    η = -1.0/vecnorm(Aq)^2.0;
    # one gradient step
    At_mul_B!(g,Aq,r)
    BLAS.axpy!(η,g,x)
end
#--------------------------------------------------------------------
# Update τ
#--------------------------------------------------------------------
function updateτ!(τ, vaq, va, m, n)

    # TODO: How to determine the step size of this function?
    # Try: one Newton step since it's smooth
    # TODO: out of box analysis
    for j = 1:n
        g = (sum(vaq[j]) - sum(va[j]))/m -
            (1.0-2.0*τ[j])/(τ[j]*(1.0-τ[j]))
        h = (2.0*τ[j]^2.0-2.0*τ[j]+1.0)/(τ[j]^2.0*(1.0-τ[j])^2.0)
        #@printf("g: %1.6e, h: %1.6e, tau: %1.3e\n",g[1],h[1],τ[1])
        τ[j] -= 0.5*g/h

        # has to keep τ in the box
        τ[j] = min(max(τ[j],0.01),0.99)
    end
end
#--------------------------------------------------------------------
# Project τ
#--------------------------------------------------------------------
function projτ!(τ, vaq, va, m, n)
    for j = 1:n
        a = (sum(va[j]) - sum(vaq[j]))/m
        a == 0.0 ? τ[j] = 0.5 :
                   τ[j] = 2.0/(sqrt(a^2+4.)-a+2.)
    end
end
#--------------------------------------------------------------------
# Update Aq
#--------------------------------------------------------------------
function updateA!(vaq, va, r, x, τ, m, n; λ=1e-3)
    # gradient step
    η = vecnorm(x)^2
    # decrease λ -> increase weight of least square term!
    for j = 1:n
        BLAS.axpy!(-x[j]/η,r,vaq[j])
        # prox-quantile
        for i = 1:m
            a = vaq[j][i] - λ*τ[j]/η
            b = a + λ/η
            vaq[j][i] = min(max(va[j][i],a),b)
        end
    end
end
#--------------------------------------------------------------------
# Calculate the Error
#   which is the normal of the subdifferential closest to 0
#--------------------------------------------------------------------
function calErr(x, vaq, va, τ, r, g, m, n)
    err = vecnorm(g)^2  # gradient from x
    for j = 1:n
        c = 0.0
        for i = 1:m
            # subgradient from Aq
            a = x[j]*r[i] + τ[j]
            b = va[j][i] - vaq[j][i]
            a < 0.0||b < 0.0 ? err += a^2.0 :
            a > 1.0||b > 0.0 ? err += (a - 1.0)^2.0 : nothing
            # linear part of τ
            c -= b
        end
        # gradient from τ
        # c += m*(-1.0/τ[j]+1.0/(1.0-τ[j]))
        # println("c: ",c)
        err += c^2
    end
    return sqrt(err)
end
#--------------------------------------------------------------------
# Main Solver
#--------------------------------------------------------------------
function tQRSolver!(x, τ, Aq, A, y; tol=1e-6, itermax=10, fixτ=false,
    pf=100)
    # pre-allocate memories
    m,n = size(A)
    r   = zeros(m)              # residual r = y-Ax
    g   = zeros(n)              # gradient for x
    # create view of A Aq and gA
    va  = [view(A,:,i) for i = 1:n]
    vaq = [view(Aq,:,i) for i = 1:n]
    AA  = zeros(n,n)
    # set iteration parameters
    err = 1.0
    num = 0
    copy!(r,y)
    BLAS.gemv!('N',1.0,Aq,x,-1.0,r)
    ob  = obj(r,vaq,va,τ,m,n)
    ob⁺ = 0.0
    while err >= tol
        num += 1;
        # proximal gradient for Aq-----------------------------------
        updateA!(vaq,va,r,x,τ,m,n)
        # project x--------------------------------------------------
        # projx!(x,AA,Aq,y)
        copy!(r,y)
        BLAS.gemv!('N',1.0,Aq,x,-1.0,r)
        updatex!(x,g,r,Aq)
        # @show(x)
        # project τ--------------------------------------------------
        fixτ || projτ!(τ,vaq,va,m,n)
        # @show(τ)
        # calculate subgradient--------------------------------------
        copy!(r,y)
        BLAS.gemv!('N',1.0,Aq,x,-1.0,r)
        err = calErr(x,vaq,va,τ,r,g,m,n)
        ob⁺ = obj(r,vaq,va,τ,m,n)
        if num % pf == 0
            @printf("iter: %6d, obj: %1.6e, err: %1.6e, res: %1.5e\n",
                num,ob⁺,err,vecnorm(r))
        end
        # if abs((ob-ob⁺)/ob) ≤ 1e-8
        #     println("iterNum: ",num)
        #     return ob⁺
        # else
        #     ob = ob⁺
        # end
        if num >= itermax
            println("Reach the Maximum Iteration!")
            break;
        end
    end
    return ob⁺
end

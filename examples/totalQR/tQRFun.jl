#====================================================================
  Total Quantile Regression Solver
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
# Project x
#--------------------------------------------------------------------
function projx!(x, AA, Aq, y)
    At_mul_B!(x,Aq,y) 
    At_mul_B!(AA,Aq,Aq)                  # A'y
    # BLAS.syrk!('U','T',1.0,Aq,0.0,AA)   # A'A
    LAPACK.sysv!('U',AA,x)              # (A'A)\(A'y)
    # At_mul_B!(x,Aq,y)
    # At_mul_B!(AA,Aq,Aq)
    # A_ldiv_B!(factorize(AA),x)
end
#--------------------------------------------------------------------
# Block Coordinate Descent for Aq
#--------------------------------------------------------------------
function BCD!(vaq, va, r, x, τ, m, n)
    for j = 1:n
        xⱼ  = x[j];    τⱼ = τ[j];       # cache to save index time
        vaⱼ = va[j]; vaqⱼ = vaq[j];
        xⱼ² = 1./xⱼ^2;
        BLAS.axpy!(xⱼ,vaq[j],r)
        for i = 1:m
        a = (r[i]*xⱼ-τⱼ)*xⱼ²
        b = a + xⱼ²
        # a > vaⱼ[i] ? vaqⱼ[i] = a :
        # b < vaⱼ[i] ? vaqⱼ[i] = b : vaqⱼ[i] = vaⱼ[i]
        vaqⱼ[i] = min(max(vaⱼ[i],a),b)
        end
        BLAS.axpy!(-xⱼ,vaq[j],r)
    end    
end
#--------------------------------------------------------------------
# Main Solver
#--------------------------------------------------------------------
function tQRSolver!(x, Aq, A, y; tol=1e-6, itermax=10)
    # pre-allocate memories
    m,n = size(A)
    τ   = fill(0.5,n)       # quantile parameter
    r   = copy(y)           # residual r = y-Ax
    AA  = zeros(n,n)        # A'A
    # create view of A Aq and gA
    va  = [view(A,:,i) for i = 1:n]
    vaq = [view(Aq,:,i) for i = 1:n]
    # set iteration parameters
    err = 1.0
    num = 0
    inn = 1
    while err >= tol
        # project x and τ
        projx!(x,AA,Aq,y)
        projτ!(τ,vaq,va,m,n)
        # block coordinate descent for Aq
        copy!(r,y)
        BLAS.gemv!('N',-1.0,Aq,xt,1.0,r)
        for k = 1:inn       # inner loop
            BCD!(vaq,va,r,x,τ,m,n)
        end
        num += 1;
        # num%10==0 &&
        #     @printf("iter %3d, obj %1.5e\n",num,obj(r,vaq,va,τ,m,n))
        if num >= itermax
            println("Reach the Maximum Iteration!")
            break;
        end
    end
    return τ
end
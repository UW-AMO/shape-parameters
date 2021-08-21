#========================================================================================
  Line Search Routines
  - Armijo line search
  - Barzilai-Borwein line search
  - Weak Wolfe line search
  - Exact line search
=========================================================================================#
#-----------------------------------------------------------------------------------------
# Armijo Line Search
# goal: find smallest k ∈ ℕ₊,s.t. f(x⁺) ≤ f(x) + α⋅c⋅⟨g,p⟩
#       where α = λᵏ and x⁺ = x + α⋅p
# input:
#   x: current point
#   f: function handle
#   g: gradient at current point, g = ∇f(x)
#   p: descent direction (need to be normalized)
#   c: sufficient descent coefficient
#   τ: shrink ratio
#   tol: lower bound of α
# output:
#   x: inplace modified in the descent direction
#   flag 0: find x⁺ before α reach tol
#   flag 1: α reach tol
#----------------------------------------------------------------------------------------
function armijo!(x, g, p, f; c = 1e-2, τ = 0.5, tol = 1e-10)
    m  = dot(g,p)
    m  ≥ 0.0 && error("Not a descent direction!")
    n  = length(x)
    α  = 1.0
    fx = f(x)
    # update x⁺
    x⁺ = copy(x)
    @simd for i = 1:n
        @inbounds x⁺[i] += α*p[i]
    end
    # start iteration
    while f(x⁺) > fx + α*c*m
        α *= τ              # update α
        @simd for i = 1:n   # update x⁺
            @inbounds x⁺[i] = x[i] + α*p[i]
        end
        # if step size is too small return the information
        if α ≤ tol
            return 1
        end
    end
    copy!(x,x⁺)
    return 0
end
#-----------------------------------------------------------------------------------------
# Barzilai-Borwein Line Search
# goal: approximate Newton step by a gradient step under certain measure, i.e.
#       xᵏ⁺¹ = xᵏ - αᵏ⋅gᵏ, use αᵏ⋅gᵏ to approximate (Hᵏ)⁻¹⋅gᵏ
# idea: set yᵏ⁻¹ = gᵏ - gᵏ⁻¹,
#           sᵏ⁻¹ = xᵏ - xᵏ⁻¹, from taylor expansion we know,
#           gᵏ⁻¹ ≈ gᵏ + Hᵏ⋅(xᵏ⁻¹ - xᵏ) ⟶ (Hᵏ)⁻¹⋅yᵏ⁻¹ ≈ sᵏ⁻¹
#       then we could formulate two problems to obtain αᵏ
#       (1) min ‖α⋅yᵏ⁻¹ - sᵏ⁻¹‖² ⟶ αᵏ = ⟨yᵏ⁻¹,sᵏ⁻¹⟩/⟨yᵏ⁻¹,yᵏ⁻¹⟩
#       (2) min ‖β⋅sᵏ⁻¹ - yᵏ⁻¹‖² ⟶ αᵏ = ⟨sᵏ⁻¹,sᵏ⁻¹⟩/⟨sᵏ⁻¹,yᵏ⁻¹⟩ = (βᵏ)⁻¹
# input:
#   x : current point
#   x⁻: previous point
#   g : gradient at current point, g = ∇f(x)
#   g⁻: gradient at previous point, g⁻ = ∇f(x⁻)
#   mean: i, use (i) to obtain α, where i ∈ {1,2}
# output:
#   x : inplace modified in the descent direction
#   x⁻: replace value by the current point
#   g⁻: replace value by the current gradient
#----------------------------------------------------------------------------------------
function bb!(x, x⁻, g, g⁻; mean = 1)
    n = length(x)
    s = zeros(n)
    y = zeros(n)
    # calculate the difference between current previous point
    @inbounds @simd for i = 1:n
        s[i] = x[i] - x⁻[i]
        y[i] = g[i] - g⁻[i]
    end
    # use different methods to calculate α
    mean == 1 ? α = dot(y,s)/dot(y,y) :
    mean == 2 ? α = dot(s,s)/dot(s,y) : error("No matching method!")
    # update all the variables
    copy!(x⁻,x)
    copy!(g⁻,g)
    BLAS.axpy!(-α,g,x)
end
#-----------------------------------------------------------------------------------------
# Weak Wolfe Line Search
# goal: try to find a step size αᵏ that satisfy the weak wolfe conditions,
#       (1) f(xᵏ + αᵏ⋅pᵏ) ≤ f(xᵏ) + c₁⋅αᵏ⋅⟨∇f(xᵏ),pᵏ⟩
#       (2) ⟨∇f(xᵏ + αᵏ⋅pᵏ),pᵏ⟩ ≥ c₂⋅⟨∇f(xᵏ),pᵏ⟩
#       where 0 < c₁ < c₂ < 1
# input:
#   x : current point
#   g : gradient at current point, g = ∇f(x)
#   p : descent direction
#   f : function handle
#   ∇f: gradient function handle
#   c₁: sufficient decrese coefficient
#   c₂: curvature coefficient
#   tol: lower bound for α
# output:
#   x : inplace modified in the descent direction
#   g : inplace modified on the new point
#   flag 0: find x⁺ before α reach tol
#   flag 1: α reach tol
#----------------------------------------------------------------------------------------
function weakWolfe!(x, g, p, f, ∇f; c₁ = 0.01, c₂ = 0.99, tol = 1e-10)
    n  = length(x)
    m  = dot(g,p)
    m  ≥ 0.0 && error("Not a descent direction!")
    α  = 1.0        # initial step size
    l  = 0.0        # left bound for bisection
    u  = Inf        # right bound for bisection
    fx = f(x)
    x⁺ = zeros(n)   # initialize next point
    @inbounds @simd for i = 1:n
        x⁺[i] = x[i] + α*p[i]
    end
    ∇f(g,x⁺)
    # start iteration
    num = 0
    while true
        num += 1
        if f(x⁺) > fx + c₁*α*m
            u = α
            α = (l+u)/2.0
        elseif dot(g,p) < c₂*m
            l = α
            u == Inf ? α = 2.0*l : (l+u)/2.0
        else
            break
        end
        # update x⁺ and g
        @inbounds @simd for i = 1:n
            x⁺[i] = x[i] + α*p[i]
        end
        ∇f(g,x⁺)
        α   < tol && (return 1)
        num ≥ 100 && (break)
    end
    copy!(x,x⁺)
    return 0
end
#-----------------------------------------------------------------------------------------
# Exact Line Search
# goal: try to find a step size αᵏ ∈ ℝ₊₊ that satisfy
#       αᵏ = argmin f(xᵏ + α⋅pᵏ)
# idea: for solving the above optimization problem, it is equivalent to solve,
#       ⟨∇f(xᵏ + α⋅pᵏ), pᵏ⟩ = 0 (when function is strictly convex)
#       here we use bisection to complish this job
# input:
#   x : current point
#   g : gradient at current point, g = ∇f(x)
#   p : descent direction
#   f : function handle
#   αmin: lower bound for α
#   tol : tolerence for bisection
# output:
#   x : inplace modified in the descent direction
#   g : inplace modified on the new point
#   flag 0: find x⁺ before α reach tol
#   flag 1: α reach tol
#   flag 2: could not find upper bound for α among {0.5,1.0,2.0,4.0,8.0}
#----------------------------------------------------------------------------------------
function exact!(x, g, p, ∇f; αmin = 1e-10, tol = 1e-8)
    n  = length(x)
    m  = dot(g,p)
    m  ≥ 0.0 && error("Not a descent direction!")
    # initialization and find the upper bound for α
    x⁺ = zeros(n)
    l  = 0.0
    u  = Inf
    for α ∈ [0.5,1.0,2.0,4.0,8.0]
        @inbounds @simd for i = 1:n
            x⁺[i] = x[i] + α*p[i]
        end
        ∇f(g,x⁺)
        m = dot(g,p)
        m  > 0.0 ? (u = α;break;) : continue
    end
    u == Inf && (return 2)
    # start bisection
    num = 0
    α   = l
    while abs(m) ≥ tol
        α = (l+u)/2.0
        α < αmin && (return 1)
        @inbounds @simd for i = 1:n
            x⁺[i] = x[i] + α*p[i]
        end
        ∇f(g,x⁺)
        m = dot(g,p)
        m > 0.0 ? u = α : l = α
        num += 1
        num >= 50 && break
    end
    copy!(x,x⁺)
    return 0
end

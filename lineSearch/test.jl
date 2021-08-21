#========================================================================================
  Test Line Search Sub-Routine On Using Poisson Regression
  Objective Function:
                    f(x) = Σ[exp(⟨aᵢ,x⟩)-bᵢ⟨aᵢ,x⟩]+λ/2⋅‖x‖²
  We will apply two algorithms
  - Gradient Descent
  - Newton's Method
=======================================================================================#
include("./lineSearch.jl");
#---------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------
function samplePoissonReg!(b, A, xt, m, n)
    BLAS.gemv!('N',1.0,A,xt,0.0,b);
    for i = 1:m
        b[i] = exp(b[i]);
        p = rand()*exp(b[i]);
        k = 0.0;
        q = b[i]^k/factorial(k);
        while p > q
            k += 1.0;
            q += b[i]^k/factorial(k);
        end
        b[i] = k;
    end
end
function f_poisson(x, A, b, λ)
    m,n = size(A);
    f   = 0.0;
    Ax  = A*x;
    for i = 1:m
        f += exp(Ax[i]) - b[i]*Ax[i];
    end
    f += 0.5*λ*vecnorm(x)^2.0
    return f
end
function g_poisson!(g, x, A, b, λ)
    m,n = size(A);
    Ax  = A*x;
    for i = 1:m
        Ax[i] = exp(Ax[i]) - b[i];
    end
    copy!(g,x)
    BLAS.gemv!('T',1.0,A,Ax,λ,g)
end
function h_poisson!(h, x, A, b, λ)
    m,n = size(A);
    Aᵀ  = A.';
    Ax  = A*x;
    for i = 1:m
        d = exp(Ax[i]);
        for j = 1:n
            Aᵀ[j,i] *= d;
        end
    end
    A_mul_B!(h,Aᵀ,A);
    for i = 1:n
        h[i,i] += λ;
    end
end
#---------------------------------------------------------------------------------------
# Generate Data
#---------------------------------------------------------------------------------------
srand(123);
m,n = 500,50;
A   = 0.3*randn(m,n);
xt  = randn(n);    # true parameters
b   = zeros(m);    # response vectors
samplePoissonReg!(b,A,xt,m,n)
λ   = 0.1;         # regularizer constant
# iteration parameters
tol = 1e-6;
itermax = 1000;
# initialization
x   = zeros(n);
g   = zeros(n);
h   = zeros(n,n);
f(y)= f_poisson(y,A,b,λ);
∇f(r,y) = g_poisson!(r,y,A,b,λ)
#---------------------------------------------------------------------------------------
# Gradient Descent
#---------------------------------------------------------------------------------------
#######################################Armijo#######################################
# num = 0;
# g_poisson!(g,x,A,b,λ);
# err = vecnorm(g);
# while err >= tol
#     info = armijo!(x,g,-g/err,f);
#     num += 1;
#     g_poisson!(g,x,A,b,λ);
#     err = vecnorm(g);
#     num % 1 == 0 && @printf("iter %4d, obj %1.5e, err %1.5e\n",num,f(x),err);
#     if info == 1
#         println("Reach Minimum Step Size!");
#         break;
#     end
#     if num ≥ itermax
#         println("Reach Maximum Iterations!");
#         break;
#     end
# end
##################################Barzilai-Borwein##################################
# # first gradient step using armijo
# num = 1;
# g_poisson!(g,x,A,b,λ);
# x⁻  = copy(x);
# g⁻  = copy(g);
# armijo!(x,f,g,-g/vecnorm(g));
# g_poisson!(g,x,A,b,λ);
# err = vecnorm(g);
# # start iteration
# while err >= tol
#     bb!(x,x⁻,g,g⁻);
#     num += 1;
#     g_poisson!(g,x,A,b,λ);
#     err = vecnorm(g);
#     num % 1 == 0 && @printf("iter %4d, obj %1.5e, err %1.5e\n",num,f(x),err);
#     if num ≥ itermax
#         println("Reach Maximum Iterations!");
#         break;
#     end
# end
#####################################Weak-Wolfe#####################################
# num = 0;
# g_poisson!(g,x,A,b,λ);
# err = vecnorm(g);
# while err >= tol
#     info = weakWolfe!(x,g,-g,f,∇f)
#     num += 1;
#     g_poisson!(g,x,A,b,λ);
#     err = vecnorm(g);
#     num % 1 == 0 && @printf("iter %4d, obj %1.5e, err %1.5e\n",num,f(x),err);
#     if num ≥ itermax
#         println("Reach Maximum Iterations!");
#         break;
#     end
#     if info == 1
#         println("Reach Minimum Step Size!");
#         break;
#     end
# end
#######################################Exact########################################
num = 0;
g_poisson!(g,x,A,b,λ);
err = vecnorm(g);
while err >= tol
    info = exact!(x,g,-g/err,∇f)
    num += 1;
    g_poisson!(g,x,A,b,λ);
    err = vecnorm(g);
    num % 1 == 0 && @printf("iter %4d, obj %1.5e, err %1.5e\n",num,f(x),err);
    if num ≥ itermax
        println("Reach Maximum Iterations!");
        break;
    end
    if info == 1
        println("Reach Minimum Step Size!");
        break;
    end
    if info == 2
        println("could not find upper bound for α among {0.5,1.0,2.0,4.0,8.0}");
        break;
    end
end
println("compare with true x: ",vecnorm(x-xt))
#---------------------------------------------------------------------------------------
# Newton's Method
#---------------------------------------------------------------------------------------
num = 0;
# initialize x, g, h, p
fill!(x,0.0);
g_poisson!(g,x,A,b,λ);
h_poisson!(h,x,A,b,λ);
p   = -h\g;
err = vecnorm(g);
while err >= tol
    info = exact!(x,g,p/vecnorm(p),∇f);
    num += 1;
    # update gradient and hessian and descent direction
    g_poisson!(g,x,A,b,λ);
    h_poisson!(h,x,A,b,λ);
    p   = -h\g;
    err = vecnorm(g);
    num % 1 == 0 && @printf("iter %4d, obj %1.5e, err %1.5e\n",num,f(x),err);
    if info == 1
        println("Reach Minimum Step Size!");
        break;
    end
    if num ≥ itermax
        println("Reach Maximum Iterations!");
        break;
    end
end
println("compare with true x: ",vecnorm(x-xt))
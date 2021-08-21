#======= Running Script for Quantile =======#

# import module and function
include("./utility.jl")
push!(LOAD_PATH,"../../src")
using PLQShape
using PyPlot

# generate data and construct PLQShape objective
srand(777)
m,n,τ,κ = 1000,50,0.1,1.0
A,xt,y = genData(m,n,[τ,κ])
@show(τ,κ)

##### Least Square Solution #####
println("-----least square solution-----")
xl = A \ y
@show norm(xl-xt)/norm(xt)

##### First Order Solver #####
println("-----first order solver-----")
# primal variable
x = zeros(n)
θ = [1.0,1.0]

# primal constraint
# S = [-1.0 1.0 0.0; 0.0 0.0 -1.0];
# s = [0.0, 1.0, 0.0];
S = -eye(2);
s = [0.0;0.0];

# primal functions
function ρ1(r,θ)
    τ = θ[1];
    κ = θ[2];
    m = length(r);
    val = 0.0;
    for i = 1:m
        r[i] <    -τ*κ ? val += -τ*κ*r[i] - 0.5*τ^2*κ^2 :
        r[i] > (1-τ)*κ ? val += (1-τ)*κ*r[i] - 0.5*(1-τ)^2*κ^2:
                         val += 0.5*r[i]^2
    end
    return val
end

function L₁(θ,A)
    return norm(A,2)^2
end

function lognc1(θ)
    τ = θ[1];
    κ = θ[2];
    val = log(exp(-0.5*(τ*κ)^2)/(τ*κ) +
        sqrt(0.5*pi)*(erf(τ*κ/sqrt(2))+erf((1-τ)*κ/sqrt(2))) +
        exp(-0.5*((1-τ)*κ)^2)/((1-τ)*κ));
    return val
end

function ρ2(r,θ)
    α = θ[1];
    β = θ[2];
    m = length(r);
    val = 0.0;
    for i = 1:m
        r[i] < -α ? val += -α*r[i] - 0.5*α^2 :
        r[i] >  β ? val +=  β*r[i] - 0.5*β^2:
                         val += 0.5*r[i]^2
    end
    return val
end

function lognc2(θ)
    a = θ[1];
    b = θ[2];
    val = log(exp(-0.5*a^2)/a +
        sqrt(0.5*pi)*(erf(a/sqrt(2))+erf(b/sqrt(2))) +
        exp(-0.5*b^2)/b);
    return val
end

# primal = PLQ_Primal(A,y,x,θ,S,s,ρ1,L₁,lognc1)

# # PALM with projection solver
# params_p = PALMProj_params(1e-6,800,1e-8,50,zeros(801))
# PALMProj(primal,params_p)
# @show θ
# @show norm(x-xt)/norm(xt)

# PALM solver
primal = PLQ_Primal(-A,-y,x,[0.5,0.5],S,s,ρ2,L₁,lognc2)
params_p = PALM_params(1e-6,800,1e-8,50,zeros(801))
θ = PALM(primal,params_p)
α = θ[1]
β = θ[2]
θ = [α/(α+β),α+β]
@show θ
@show norm(x-xt)/norm(xt)

##### Second Order Solver #####
println("-----second order solver-----")
# primal variables


# dual variables
B = eye(m)
G = zeros(2,m)
b = zeros(m)
M = eye(m)

C = [-eye(m) eye(m)]
H = [ones(1,m) zeros(1,m);zeros(1,m) ones(1,m)]
c = zeros(2*m)

primal = PLQ_Primal(-A,-y,x,[0.5,0.5],S,s,ρ2,L₁,lognc2)
dual   = PLQ_Dual(B,G,b,M,C,H,c)


# IPsolve solver
params_i = IP_params(1.0,0.1,1e-6,50,zeros(51))
x,θ = IPsolve(primal,dual,params_i)
α = θ[1]
β = θ[2]
θ = [α/(α+β),α+β]
@show θ
@show norm(x-xt)/norm(xt)

# plot convergent history
optval = params_i.convergent_history[end]
# PLAM only
# figure()
# semilogy(params_p.convergent_history[1:100] - optval,"-g")
# xlabel("iterations")
# ylabel("obj val - optimal obj val")
# legend(["PALM with projection"])
# savefig("conHis_PALM.eps", transparent=true)

# both IPsolve and PLAM
figure()
semilogy(params_p.convergent_history[1:100] - optval,"-g", linewidth=3)
semilogy(params_i.convergent_history - optval,"-b", linewidth=3)
# title("Convergent History")
xlabel("iterations",fontsize=20)
ylabel(L"$f - f^*$",fontsize=25)
# legend(["PALM variation","IPsolve"])
savefig("conHis_both.eps", transparent=true)
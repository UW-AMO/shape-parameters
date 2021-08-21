#====================================================================
  Total Quantile Regression
  -Load Data
  -Least Square
  -Total Quantile
  -Compare Result
====================================================================#
#--------------------------------------------------------------------
# Generate Data
#--------------------------------------------------------------------
include("data/genData.jl")
m,n = 500,2
A, y, At, xt, τt = genData(m,n)
#--------------------------------------------------------------------
# Least Square
#--------------------------------------------------------------------
x₁  = A\y
er1 = vecnorm(x₁-xt)/vecnorm(xt)
@printf("Least Square Err: %1.5e\n",er1)
#--------------------------------------------------------------------
# Total Quantile
#--------------------------------------------------------------------
include("./tQRPALM.jl")
# τ   = fill(0.5,n)
τ   = copy(τt)
x₂  = ones(n)
# x₂  = copy(xt)
Aq  = copy(At)
# @show(size(Aq))
tQRSolver!(x₂,τ,Aq,A,y,itermax=800000,fixτ=false,pf=10000)
er2 = vecnorm(x₂[1:n]-xt)/vecnorm(xt)
@printf("Total Quantile Err: %1.5e\n",er2)
println("Difference in τt: ",τt, ", τ:",τ)
# println("Difference in xt: ",xt, ", x:",x₂)
# println("Difference in At: ",At[10:20], ", A:",Aq[10:20])
# @show(sum(xt))
# @show(sum(x₂))
#--------------------------------------------------------------------
# Compare the True Objective Value
#--------------------------------------------------------------------
# rt  = copy(y)
# BLAS.gemv!('N',1.0,At,xt,-1.0,rt)
# va  = [view(A,:,i) for i = 1:n]
# vat = [view(At,:,i) for i = 1:n]
# println("true objective value ",obj(rt,vat,va,τt,m,n))
# r   = copy(y)
# BLAS.gemv!('N',1.0,Aq,x₂,-1.0,r)
# @show(vecnorm(rt)^2)
# @show(vecnorm(r)^2)

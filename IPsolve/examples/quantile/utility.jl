#====================================================================
    Utility Function for
    - generate data
====================================================================#

##### inverse CDF of quantile distribution #####
function icdfQ(p::Float64,τ::Float64)
    p == 0.0 && (return -Inf)
    p == 1.0 && (return  Inf)
    p < (1.0-τ) ? val = log(p/(1.0-τ))/τ :
                  val = log((1.0-p)/τ)/(τ-1.0);
    return val
end

##### sample from quantile distribution (scalar case) #####
function sampleQ(τ::Float64)
    p = rand();
    return icdfQ(p,τ)
end

##### data generater #####
function genData(m::Int64,n::Int64,τ::Float64)
    A = randn(m,n)
    x = randn(n)
    y = A*x
    for i = 1:m
        y[i] -= sampleQ(τ)
    end
    return A, x, y
end
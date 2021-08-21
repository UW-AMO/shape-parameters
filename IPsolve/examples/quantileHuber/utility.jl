#= ===================================================================
    Utility Function for
    - generate data
=================================================================== =#
using SpecialFunctions:erf

##### quantile Huber penalty #####
function quantileHuber(x::Float64, τ::Float64, κ::Float64)
    # parameter domain check
    κ > 0 || error("κ must be a positive number!");
    τ > 0 && τ < 1 || error("τ must between 0 and 1!");
    # apply penalty
    x < -τ * κ    ? val =    -τ * κ * x - 0.5 * (τ * κ)^2     :
    x > (1 - τ) * κ ? val = (1 - τ) * κ * x - 0.5 * ((1 - τ) * κ)^2 :
                  val = 0.5 * x^2;
    return val
end

##### CDF of quantile Huber distribution #####
function cdfQH(x::Float64, τ::Float64, κ::Float64, c::Float64)
    # we fully use closed form here
    if x < -τ * κ
        p = exp(τ * κ * (x + 0.5 * τ * κ)) / (τ * κ);
    elseif x > (1 - τ) * κ
        p = exp(-0.5 * (τ * κ)^2) / (τ * κ) +
            sqrt(0.5 * pi) * (erf(τ * κ / sqrt(2)) + erf((1 - τ) * κ / sqrt(2))) -
            exp((1 - τ) * κ * (0.5 * (1 - τ) * κ - x)) / ((1 - τ) * κ) +
            exp(-0.5 * ((1 - τ) * κ)^2) / ((1 - τ) * κ);
    else
        p = exp(-0.5 * (τ * κ)^2) / (τ * κ) +
            sqrt(0.5 * pi) * (erf(τ * κ / sqrt(2)) + erf(x / sqrt(2)));
    end
    return p / c
end

##### inverse CDF of quantile Huber distribution #####
function icdfQH(p::Float64, θ::Array{Float64,1})
    τ = θ[1];
    κ = θ[2];
    # normalization constant
    c = exp(-0.5 * (τ * κ)^2) / (τ * κ) +
        sqrt(0.5 * pi) * (erf(τ * κ / sqrt(2)) + erf((1 - τ) * κ / sqrt(2))) +
    exp(-0.5 * ((1 - τ) * κ)^2) / ((1 - τ) * κ);
    tol = 1e-8;
    iter = 0;
    itermax = 100;
    x = 0.0;
        err = p - cdfQH(x, τ, κ, c);
    while abs(err) >= tol
        x += err * c / exp(-quantileHuber(x, τ, κ))
        err = p - cdfQH(x, τ, κ, c);
        iter += 1
        if iter >= itermax
            print("Reach Maximum Iteration!\n");
            break;
        end
    end
    return x
end

##### sample from quantile Huber distribution (scalar case) #####
function sampleQH(θ::Array{Float64,1})
    p = rand();
    return icdfQH(p, θ)
end

##### data generater #####
function genData(m::Int64, n::Int64, θ::Array{Float64,1})
    A   = randn(m, n)
    # A, _ = qr(A)
    x   = randn(n)
    y   = A * x 
    for i = 1:m
        y[i] += sampleQH(θ)
    end
    return A, x, y
end
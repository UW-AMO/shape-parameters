#====================================================================
    Utility Function for
    - generate data
====================================================================#

##### Huber penalty #####
function huber(x::Float64,κ::Float64)
    # parameter domain check
    κ > 0 || error("κ must be a positive number!");
    # apply penalty
    x < -κ ? val = -κ*x-0.5*κ^2 :
    x >  κ ? val =  κ*x-0.5*κ^2 :
                  val = 0.5*x^2;
    return val
end

##### CDF of Huber distribution #####
function cdfH(x::Float64,κ::Float64,c::Float64)
    # we fully use closed form here
    if x < -κ
        p = exp(κ*(x+0.5*κ))/κ;
    elseif x > κ
        p = exp(-0.5*κ^2)/κ +
            sqrt(0.5*pi)*(erf(κ/sqrt(2))+erf(κ/sqrt(2))) -
            exp(κ*(0.5*κ-x))/κ +
            exp(-0.5*κ^2)/κ;
    else
        p = exp(-0.5*κ^2)/κ +
            sqrt(0.5*pi)*(erf(κ/sqrt(2))+erf(x/sqrt(2)));
    end
    return p/c
end

##### inverse CDF of quantile Huber distribution #####
function icdfH(p::Float64,κ::Float64)
    # normalization constant
    c = exp(-0.5*κ^2)/κ +
        sqrt(0.5*pi)*(erf(κ/sqrt(2))+erf(κ/sqrt(2))) +
        exp(-0.5*κ^2)/κ;
    tol = 1e-8;
    iter = 0;
    itermax = 100;
    x = 0.0;
    err = p-cdfH(x,κ,c);
    while abs(err) >= tol
        x += err*c/exp(-huber(x,κ))
        err = p-cdfH(x,κ,c);
        iter += 1
        if iter >= itermax
            print("Reach Maximum Iteration!\n");
            break;
        end
    end
    return x
end

##### sample from quantile Huber distribution (scalar case) #####
function sampleH(κ::Float64)
    p = rand();
    return icdfH(p,κ)
end

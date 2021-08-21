#====================================================================
  Sampler for Distribution Generated by Quantile-Huber Penalty
  - quantile-huber penalty
  - pdf of distribution
  - cdf of distribution
  - inverse cdf of distribution
  - sample from this distribution
====================================================================#
#--------------------------------------------------------------------
# Quantile-Huber Penalty
#--------------------------------------------------------------------
function quantileHuber(x::Float64,τ::Float64,κ::Float64)
	# parameter domain check
	κ > 0 || error("κ must be a positive number!");
	τ > 0 && τ < 1 || error("τ must between 0 and 1!");
	# apply penalty
	x < -τ*κ    ? val =    -τ*κ*x-0.5*(τ*κ)^2     :
	x > (1-τ)*κ ? val = (1-τ)*κ*x-0.5*((1-τ)*κ)^2 :
				  val = 0.5*x^2;
	return val
end
#--------------------------------------------------------------------
# PDF of Distribution
#--------------------------------------------------------------------
function pdfQH(x::Float64,τ::Float64,κ::Float64)
	# normalization constant
	c = exp(-0.5*(τ*κ)^2)/(τ*κ) +
		sqrt(0.5*pi)*(erf(τ*κ/sqrt(2))+erf((1-τ)*κ/sqrt(2))) +
		exp(-0.5*((1-τ)*κ)^2)/((1-τ)*κ);
	# could also use numerical evaluation
	# f(x) = exp(-quantileHuber(x,τ,κ));
	# c, e = quadgk(f,-Inf,Inf);
	return exp(-quantileHuber(x,τ,κ))/c
end
#--------------------------------------------------------------------
# CDF of Distribution
#--------------------------------------------------------------------
function cdfQH(x::Float64,τ::Float64,κ::Float64)
	# we fully use numerical method here
	f(x) = exp(-quantileHuber(x,τ,κ));
	c, e = quadgk(f,-Inf,Inf);
	p, e = quadgk(f,-Inf,x);
	return p/c
end
function cdfQH(x::Float64,τ::Float64,κ::Float64,c::Float64)
	# we fully use closed form here
	if x < -τ*κ
		p = exp(τ*κ*(x+0.5*τ*κ))/(τ*κ);
	elseif x > (1-τ)*κ
		p = exp(-0.5*(τ*κ)^2)/(τ*κ) +
			sqrt(0.5*pi)*(erf(τ*κ/sqrt(2))+erf((1-τ)*κ/sqrt(2))) -
			exp((1-τ)*κ*(0.5*(1-τ)*κ-x))/((1-τ)*κ) +
			exp(-0.5*((1-τ)*κ)^2)/((1-τ)*κ);
	else
		p = exp(-0.5*(τ*κ)^2)/(τ*κ) +
			sqrt(0.5*pi)*(erf(τ*κ/sqrt(2))+erf(x/sqrt(2)));
	end
	return p/c
end
#--------------------------------------------------------------------
# Inverse CDF of Distribution
#--------------------------------------------------------------------
function icdfQH(p::Float64,τ::Float64,κ::Float64)
	# normalization constant
	c = exp(-0.5*(τ*κ)^2)/(τ*κ) +
		sqrt(0.5*pi)*(erf(τ*κ/sqrt(2))+erf((1-τ)*κ/sqrt(2))) +
		exp(-0.5*((1-τ)*κ)^2)/((1-τ)*κ);
	tol = 1e-8;
	iter = 0;
	itermax = 100;
	x = 0.0;
	err = p-cdfQH(x,τ,κ,c);
	while abs(err) >= tol
		x += err*c/exp(-quantileHuber(x,τ,κ))
		err = p-cdfQH(x,τ,κ,c);
		iter += 1
		if iter >= itermax
			print("Reach Maximum Iteration!\n");
			break;
		end
	end
	return x
end
#--------------------------------------------------------------------
# Sample Function for Distribution (only scalar case)
#--------------------------------------------------------------------
function sampleQH(τ::Float64,κ::Float64)
	p = rand();
	return icdfQH(p,τ,κ)
end
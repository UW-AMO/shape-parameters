#====================================================================
 Different Function Used for Maximum Likelihood
 - ml objective value
 no need - penalty derivatives w.r.t. parameters
 no need - gradient and hessian for the ml objective
 - Newton solver for the optimization problem
====================================================================#
using ForwardDiff
#--------------------------------------------------------------------
# Objective Value
#--------------------------------------------------------------------
function obj(r::Array{Float64,1},τ,κ)
	m   = length(r);
	val = 0.0;
	for i = 1:m
		r[i]<   -τ*κ ? val+=    -τ*κ*r[i]-0.5*(τ*κ)^2 :
		r[i]>(1-τ)*κ ? val+= (1-τ)*κ*r[i]-0.5*((1-τ)*κ)^2 :
					   val+= 0.5*r[i]^2;
	end
	val /= m;
	# print("τ, κ:",τ," ",κ,"\n")
	val += log(exp(-0.5*(τ*κ)^2)/(τ*κ) +
		sqrt(0.5*pi)*(erf(τ*κ/sqrt(2))+erf((1-τ)*κ/sqrt(2))) +
		exp(-0.5*((1-τ)*κ)^2)/((1-τ)*κ));
	return val
end
#--------------------------------------------------------------------
# Newton Solver
# TODO: Add Barrier Function to Keep κ and τ Feasible
#--------------------------------------------------------------------
function ntSolver(τ::Float64,κ::Float64,r::Array{Float64,1},
				tol::Float64,itermax::Int64)
	f(x) = obj(r,x[1],x[2]);
	μ = 0.0;		# constant for barrier function
	g = Array(Float64,2);
	h = Array(Float64,2,2);
	ForwardDiff.gradient!(g,f,[τ,κ]);
	ForwardDiff.hessian!(h,f,[τ,κ]);
	g[1] -= μ*(1.0/τ + 1.0/(τ-1));
	g[2] -= μ/κ;
	h[1] += μ*(1.0/τ^2+1.0/(τ-1)^2);
	h[4] += μ/κ^2;
	err = norm(g);
	num = 0;
	while err >= tol
		p   = h\g;
		τ  -= p[1];
		κ  -= p[2];
		ForwardDiff.gradient!(g,f,[τ,κ]);
		ForwardDiff.hessian!(h,f,[τ,κ]);
		g[1] -= μ*(1.0/τ + 1.0/(τ-1));
		g[2] -= μ/κ;
		h[1] += μ*(1.0/τ^2+1.0/(τ-1)^2);
		h[4] += μ/κ^2;
		μ  /= 10;
		err = max(norm(g),μ);
		num += 1;
		if num >= itermax
			print("Reach Maximum Iteration!\n");
			break;
		end
	end
	return τ, κ
end
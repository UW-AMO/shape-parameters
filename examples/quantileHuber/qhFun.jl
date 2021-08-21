#====================================================================
 Different Functions About Quantile-Huber Objective
 funv: function value of quantile-huber
 qhThresh!: quatile-huber threshhold
 grad!: gradient of Quantile-Huber Objective
 BFGSSolver: implementing BFGS method
====================================================================#
import Roots.fzero;
using ForwardDiff;
#--------------------------------------------------------------------
# Function Value
#--------------------------------------------------------------------
function funv(r::Array{Float64,1},τ::Float64,κ::Float64)
	val = 0.0;
	m   = length(r);
	for i = 1:m
		r[i]<   -τ*κ ? val+=    -τ*κ*r[i]-0.5*(τ*κ)^2 :
		r[i]>(1-τ)*κ ? val+= (1-τ)*κ*r[i]-0.5*((1-τ)*κ)^2 :
					   val+= 0.5*r[i]^2;
	end
	return val
end
function funv(x,y::Array{Float64,1},A::Array{Float64,2},
			τ::Float64,κ::Float64)
	r   = A*x - y;
	val = 0.0;
	m   = length(r);
	for i = 1:m
		r[i]<   -τ*κ ? val+=    -τ*κ*r[i]-0.5*(τ*κ)^2 :
		r[i]>(1-τ)*κ ? val+= (1-τ)*κ*r[i]-0.5*((1-τ)*κ)^2 :
					   val+= 0.5*r[i]^2;
	end
	return val
end
#--------------------------------------------------------------------
# Quantile-Huber Threshhold
#--------------------------------------------------------------------
function qhThresh(r::Array{Float64},τ::Float64,κ::Float64)
	m  = length(r);
	rs = zeros(m);
	for i = 1:m
		r[i]<   -τ*κ ? rs[i]=   -τ*κ :
		r[i]>(1-τ)*κ ? rs[i]=(1-τ)*κ : rs[i]=r[i];
	end
	return rs
end
function qhThresh!(rs::Array{Float64,1},r::Array{Float64,1},
					τ::Float64,κ::Float64)
	m  = length(r);
	for i = 1:m
		r[i]<   -τ*κ ? rs[i]=   -τ*κ :
		r[i]>(1-τ)*κ ? rs[i]=(1-τ)*κ : rs[i]=r[i];
	end
end
#--------------------------------------------------------------------
# Gradient
#--------------------------------------------------------------------
function grad!(g::Array{Float64,1},r::Array{Float64,1},
	rs::Array{Float64,1},A::Array{Float64,2},τ::Float64,κ::Float64)
	# n  = length(g);
	qhThresh!(rs,r,τ,κ);
	BLAS.gemv!('T',1.0,A,rs,0.0,g);
	# for i = 1:n g[i] = dot(A[:,i],rs); end
end
#--------------------------------------------------------------------
# My Root Solver
#--------------------------------------------------------------------
function fg(α::Float64,r::Array{Float64,1},d::Array{Float64,1},
			τ::Float64,κ::Float64)
	f = 0.0;
    g = 0.0;
    for i = 1:m
        if r[i]+α*d[i] < -τ*κ
            f += -τ*κ*d[i];
        elseif r[i]+α*d[i] > (1-τ)*κ
            f += (1-τ)*κ*d[i];
        else
            f += (r[i]+α*d[i])*d[i];
            g += d[i]^2;
        end
    end
    # g==0.0 ? g = 1.0 : nothing;
    return f,g
end
function ntRoot(x::Float64,r::Array{Float64,1},d::Array{Float64,1},
				τ::Float64,κ::Float64)
    f,g = fg(x,r,d,τ,κ);
    print("f,g: ",f," ",g,"\n");
    err = abs(f);
    num = 0;
    while err >= 1e-6
        x -= f/g;
        f,g = fg(x,r,d,τ,κ);
        err = abs(f);
        num += 1;
        if num > 100
            print("Reach Maximum Iteration!\n");
            break;
        end
    end
    return x
end
#--------------------------------------------------------------------
# BFGS Solver
#--------------------------------------------------------------------
function BFGSSolver!(y::Array{Float64,1},A::Array{Float64,2},
					 x::Array{Float64,1},τ::Float64,κ::Float64,
					 tol::Float64,itermax::Int64)
	# initialize all the parameters
	m,n = size(A);
	# r = A*x - y;
	r   = zeros(m);
	BLAS.gemv!('N',1.0,A,x,0.0,r);
	for i = 1:m r[i] -= y[i]; end
	rs  = zeros(m);
	g   = zeros(n);
	w   = zeros(n);
	Hw  = zeros(n);
	p   = zeros(n);
	d   = zeros(m);
	# qh(x) = funv(x,y,A,τ,κ);
	# ForwardDiff.gradient!(g,qh,x);
	grad!(g,r,rs,A,τ,κ);
	H   = eye(n);
	err = norm(g);
	num = 0;
	while err >= tol
		# compute search direction
		# p    = -H*g;
		BLAS.gemv!('N',-1.0,H,g,0.0,p);
		# line search
		# d    = A*p;
		BLAS.gemv!('N',1.0,A,p,0.0,d);
		function f(α)
			v = 0.0
			for i = 1:m
				r[i]+α*d[i] < -τ*κ ? 	v += -τ*κ*d[i] :
				r[i]+α*d[i] > (1-τ)*κ ? v += (1-τ)*κ*d[i] :
										v += (r[i]+α*d[i])*d[i];
			end
			return v
		end
		α = fzero(f,[0.0,40.0]);
		# α = ntRoot(0.0,r,d,τ,κ);
		# update parameters
		# s    = α*p;
		for i = 1:n x[i] += α*p[i]; end
		BLAS.gemv!('N',1.0,A,x,0.0,r);
		for i = 1:m r[i] -= y[i]; end
		# ForwardDiff.gradient!(w,qh,x);
		grad!(w,r,rs,A,τ,κ);
		for i = 1:n w[i] -= g[i]; end
		# Hw   = H*w;
		BLAS.gemv!('N',1.0,H,w,0.0,Hw);
		wHw  = dot(w,Hw);
		ρ    = 1.0/dot(w,p);
		c    = α*ρ+ρ^2*wHw;
		for i = 1:n
		for j = 1:n
			H[i,j] -= ρ*(Hw[i]*p[j]+p[i]*Hw[j])-c*p[i]*p[j];
		end
		end
		for i = 1:n g[i] += w[i]; end
		err  = norm(g);
		num += 1;
		# @printf("iter %2i, val %1.5e, err %1.5e, step %1.5e\n",
		# 	num,funv(r,τ,κ),err,α);
		if num >= itermax
			print("Reach Maximum Iteration!\n");
			break;
		end
	end
	# print(num,"\n");
end
#====================================================================
 Different Functions About Huber Objective
 sofT: soft threshold of a vector
 grad: gradient of huber objective
 gdSolver: gradient descend solver for huber objective
 BFGSSolver: implementing BFGS method
====================================================================#
using Optim;
import Roots.fzero;
#--------------------------------------------------------------------
# Soft Threshold Function
#--------------------------------------------------------------------
function sofT!(x::Array{Float64,1},κ::Float64)
	m = length(x);
	for i = 1:m
		x[i]> κ?x[i]= κ:
		x[i]<-κ?x[i]=-κ:0;
	end
end
function soft(r::Array{Float64,1},κ::Float64)
	m = length(r);
	s = copy(r);
	for i = 1:m
		abs(s[i])>κ ? s[i] = κ*sign(s[i]) : 0;
	end
	return s
end
#--------------------------------------------------------------------
# Gradient
#--------------------------------------------------------------------
function grad(y::Array{Float64,1},A::Array{Float64,2},
			  x::Array{Float64,1},κ::Float64)
	g = A*x - y;
	sofT!(g,κ);
	g = A'*g;
	return g
end
#--------------------------------------------------------------------
# Gradient Descend Solver
#--------------------------------------------------------------------
function gdSolver!(y::Array{Float64,1},A::Array{Float64,2},
				   x::Array{Float64,1},κ::Float64,η::Float64,
				   tol::Float64,itermax::Int64)
	n    = length(x);
	g    = grad(y,A,x,κ);
	α    = η;
	err  = norm(g);
	iter = 0;
	while err >= tol
		# update x
		for i = 1:n x[i] -= α*g[i]; end
		# update grdient
		iter  += 1;
		if iter % 10000 == 0
			@printf("xiter %d, err %1.5e\n",iter,err);
		end
		if iter > itermax
			print("Warning Reach Maximum Iter Number!\n");
			break;
		end
		g   = grad(y,A,x,κ);
		err = norm(g);
	end
	# @printf("xiter %d, err %1.5e\n",iter,err);
end
#--------------------------------------------------------------------
# BFGS Solver
#--------------------------------------------------------------------
function BFGSSolver!(y::Array{Float64,1},A::Array{Float64,2},
					x::Array{Float64,1},κ::Float64,
					tol::Float64,itermax::Int64)
	r = A*x - y;
	g = grad(y,A,x,κ);
	H = eye(n);
	err = norm(g);
	num = 0;
	while err >= tol
		# compute search direction
		p    = -H*g;
		# line search (solve (Ap)'soft(r+Ap)==0)
		d    = A*p;
		f(α) = dot(d,soft(r+α*d,κ));
		α    = fzero(f,[0.0,10.0]);
		# update parameters
		s    = α*p;
		for i = 1:n x[i] += s[i]; end
		r    = A*x - y;
		gnew = grad(y,A,x,κ);
		w    = gnew - g;
		Hw   = H*w;
		wHw  = dot(w,Hw);
		ρ    = 1.0/dot(w,s);
		# H    = H - ρ*(Hw*s'+s*Hw')+(ρ+ρ^2*wHw)*s*s';
		c    = ρ+ρ^2*wHw;
		for i = 1:n
		for j = 1:n
			H[i,j] -= ρ*(Hw[i]*s[j]+s[i]*Hw[j])-c*s[i]*s[j];
		end
		end
		g    = gnew;
		err  = norm(g);
		num += 1
		# @printf("iter %4i, obj %1.5e, step %1.5e\n",
		# 	num,func(r,κ),α);
		if num >= itermax
			print("Warning: Reach Max Iterations!\n");
			break;
		end
	end
end
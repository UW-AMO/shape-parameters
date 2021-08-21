#====================================================================
 Different Functions About Quantile Objective
 funv: function value of quantile-huber
 conj: conjuate function w.r.t. x
 prox_conj!: prox operator of conjugate function
 pdSolver!: first order primal-dual solver
 subBFGS!: BFGS method for nonsmooth objective
====================================================================#
import Roots.fzero
using ForwardDiff
#--------------------------------------------------------------------
# Function Value
#--------------------------------------------------------------------
function funv(x::Float64,τ::Float64)
	x < 0.0 ? val = -τ*x : val = (1-τ)*x;
	return val
end
function funv(x,τ::Float64)
	m   = length(x);
	val = 0.0;
	for i = 1:m
		x[i] < 0.0 ? val += -τ*x[i] : val += (1-τ)*x[i];
	end
	return val
end
#--------------------------------------------------------------------
# Conjugate Function
#--------------------------------------------------------------------
function conj(y::Float64,τ::Float64)
	-τ <= y <= (1-τ) ? val = 0.0 : val = Inf;
	return val
end
function conj(y::Array{Float64,1},τ::Float64)
	m   = length(y);
	for i = 1:m
		if y[i] < -τ || y[i] > 1-τ
			return Inf
		end
	end
	return 0.0
end
#--------------------------------------------------------------------
# Prox Operator of Conjuate Function
#--------------------------------------------------------------------
function prox_conj!(p::Array{Float64,1},w::Array{Float64,1},
					τ::Float64)
	m = length(w);
	for i = 1:m
		w[i] <  -τ ? p[i] =  -τ :
		w[i] > 1-τ ? p[i] = 1-τ : p[i] = w[i];
	end
end
#--------------------------------------------------------------------
# First Order Primal-Dual Solver
#--------------------------------------------------------------------
function pdSolver!(x::Array{Float64,1},A::Array{Float64,2},
				   b::Array{Float64,1},τ::Float64,
				   tol::Float64,itermax::Int64)
	m,n = size(A);
	# allocate memories for all the variables
	y   = Array(Float64,m);
	rp  = Array(Float64,m);
	rd  = Array(Float64,n);
	dr  = Array(Float64,m);
	# initialize variables
	λ   = 2e-04; # DON'T KNOW HOW TO CHOOSE THIS
	# rp = A*x - b
	BLAS.gemv!('N',1.0,A,x,0.0,rp);
	for i = 1:m rp[i] -= b[i]; end
	# y = ρ'(rp)
	for i = 1:m y[i] = λ*rp[i]; end
	prox_conj!(y,y,τ);
	# rd = A^T*y
	BLAS.gemv!('T',1.0,A,y,0.0,rd);
	# start iteration
	err = 1.0;
	num = 0;
	while err >= tol
		# xp = x - λ*rd
		for i = 1:n x[i] -= λ*rd[i]; end
		# dr = A*(xp-x)/λ = -A*rd
		BLAS.gemv!('N',-1.0,A,rd,0.0,dr);
		# rp = A*x - b
		BLAS.gemv!('N',1.0,A,x,0.0,rp);
		for i = 1:m rp[i] -= b[i]; end
		# y = y + λ^2*dr + λ*rp
		for i = 1:m y[i] += λ^2*dr[i] + λ*rp[i]; end
		# prox step
		prox_conj!(y,y,τ);
		# rd = A^T*y
		BLAS.gemv!('T',1.0,A,y,0.0,rd);
		err = norm(rd);
		num += 1;
		if num % 1000 == 0
			@printf("Iter %4d, error %1.5e, obj %1.5e\n",
					num, err, funv(rp,τ));
		end
		if num >= itermax
			print("Reach Maximum Iteration!\n");
			break;
		end
	end
end

# function pdSolver!(x::Array{Float64,1},A::Array{Float64,2},
# 				   b::Array{Float64,1},τ::Float64,
# 				   tol::Float64,itermax::Int64)
# 	m,n = size(A);
# 	y   = zeros(m);
# 	xp  = copy(x);
# 	err = 1.0;
# 	num = 0;
# 	λ   = 1e-04;
# 	while err >= tol
# 		y = y + λ*A*xp-λ*b;
# 		prox_conj!(y,y,τ);
# 		xnew = x - λ*A'*y;
# 		xp = 2*xnew - x;
# 		for i = 1:n x[i] = xnew[i]; end
# 		err = norm(A'*y);
# 		num += 1;
# 		if num % 100 == 0
# 			@printf("Iter %4d, error %1.5e, obj %1.5e\n",
# 					num, err, funv(A*x-b,τ));
# 		end
# 		if num >= itermax
# 			print("Reach Maximum Iteration!\n");
# 			break;
# 		end
# 	end
# end
#--------------------------------------------------------------------
# Sub BFGS Solver
#--------------------------------------------------------------------
function grad!(g::Array{Float64,1},r::Array{Float64,1},
			   rs::Array{Float64,1},A::Array{Float64,2},
			   H::Array{Float64,2},τ::Float64)
	m,n = size(A);
	idz = 0;
	for i = 1:m
		r[i] > 0.0 ? rs[i] = 1-τ : rs[i] = -τ;
		if abs(r[i]) <= 1e-15
			rs[i] = 0.0
			idz   = i;
		end
	end
	BLAS.gemv!('T',1.0,A,rs,0.0,g);
	# print("number of zero ",noz,"\n");
	# only choose one dimension to do the sub problem find the grad
	# if idz != 0
	# 	a  = A'[:,idz];
	# 	Ha = zeros(n);
	# 	BLAS.gemv!('N',1.0,H,a,0.0,Ha);
	# 	x  = -dot(g,Ha)/dot(a,Ha);
	# 	x < -τ   ? rs[idz] = -τ  :
	# 	x >(1-τ) ? rs[idz] = 1-τ : rs[idz] = x;
	# 	# print("rs[idz] ",rs[idz],"\n");
	# 	BLAS.gemv!('T',1.0,A,rs,0.0,g);
	# end
end
function linesearch(r::Array{Float64,1},d::Array{Float64,1},
					id::Array{Int64,1},zp::Array{Float64,1},τ::Float64)
	m = length(r);
	v = 0.0;
	k = 0
	for i = 1:m
		if d[i] != 0.0
			if r[i]*d[i] > 0.0
				d[i] > 0 ? v += d[i]*(1-τ) : v -= d[i]*τ;
			else
				k += 1;
				id[k] = i;
				zp[k] = -r[i]/d[i];
				d[i] < 0 ? v += d[i]*(1-τ) : v += d[i]*(-τ);
			end
		end
	end
	p = sortperm(zp[1:k]);
	j = p[1];
	if v >= 0
		return 0.0
	else
		for i = 1:k
			j = p[i];
			v += abs(d[id[j]]);
			if v >= 0
				break;
			end
		end
		return zp[j]
	end
end
function subBFGS!(x::Array{Float64,1},A::Array{Float64,2},
				  y::Array{Float64,1},τ::Float64,
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
	id  = Array(Int64,m);
	zp  = zeros(m);
	# qh(x) = funv(x,y,A,τ,κ);
	# ForwardDiff.gradient!(g,qh,x);
	H   = eye(n);
	grad!(g,r,rs,A,H,τ);
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
				r[i]+α*d[i] < 0.0 ? v += -τ*d[i]    :
				r[i]+α*d[i] > 0.0 ? v += (1-τ)*d[i] : nothing;
			end
			return v
		end
		# @printf("f(0) %1.5e, f(100) %1.5e\n",f(0.0),f(100.0))
		# α = fzero(f,[0.0,1.0]);
		α = linesearch(r,d,id,zp,τ);
		if abs(α) <= 1e-30
			print("Not a descent direction\n");
			break;
		end
		# update parameters
		# s    = α*p;
		for i = 1:n x[i] += α*p[i]; end
		# BLAS.gemv!('N',1.0,A,x,0.0,r);
		for i = 1:m r[i] += α*d[i]; end
		# ForwardDiff.gradient!(w,qh,x);
		grad!(w,r,rs,A,H,τ);
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
		# num%10==0 && @printf("iter %3i, val %1.5e, err %1.5e, step %1.5e\n",
		# 	num,funv(r,τ),err,α);
		if num >= itermax
			print("Reach Maximum Iteration!\n");
			break;
		end
	end
	# print(num,"\n");
end
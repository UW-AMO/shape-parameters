#====================================================================
 Functions and Solver About Projected Quantile-Huber
 pGrad!: gradient of projected Quantile-Huber
 pSolver: implementing BFGS method for projected QH
====================================================================#
# include("./qhFun.jl");
# include("./mlFun.jl");
#--------------------------------------------------------------------
# Function Value for Projected Quantile-Huber
#--------------------------------------------------------------------
function pVal(r::Array{Float64,1})
	τ,κ = ntSolver(0.5,0.1,r,1e-6,100);
	return funv(r,τ,κ)
end
#--------------------------------------------------------------------
# Gradient for Projected Quantile-Huber
#--------------------------------------------------------------------
function pGrad!(g::Array{Float64,1},r::Array{Float64,1},
				rs::Array{Float64,1},A::Array{Float64,2})
	τ,κ = ntSolver(0.5,0.1,r,1e-6,100);
	# print("τ: ",τ,", κ:",κ,"\n");
	grad!(g,r,rs,A,τ,κ);
	return τ,κ
end
#--------------------------------------------------------------------
# BFGS Solver for projected QH
#--------------------------------------------------------------------
function pSolver!(y::Array{Float64,1},A::Array{Float64,2},
				x::Array{Float64,1},tol::Float64,itermax::Int64)
	# pre-calculate
	BFGSSolver!(y,A,x,0.01,0.1,tol,itermax);
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
	τ,κ = pGrad!(g,r,rs,A);
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
		# τ,κ = ntSolver(0.01,0.1,r,1e-6,100);
		function linef(α)
			# print("α: ",α,"\n");
			v = 0.0;
			for i = 1:m
				r[i]+α*d[i] < -τ*κ ? 	v += -τ*κ*d[i] :
				r[i]+α*d[i] > (1-τ)*κ ? v += (1-τ)*κ*d[i] :
										v += (r[i]+α*d[i])*d[i];
			end
			return v
		end
		# print("this point\n");
		α = fzero(linef,[0.0,20.0]);
		# print("that point\n");
		# α = ntRoot(0.0,r,d,τ,κ);
		# update parameters
		# s    = α*p;
		for i = 1:n x[i] += α*p[i]; end
		BLAS.gemv!('N',1.0,A,x,0.0,r);
		for i = 1:m r[i] -= y[i]; end
		# ForwardDiff.gradient!(w,qh,x);
		τ,κ = pGrad!(w,r,rs,A);
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
		if num % 10 == 0
			@printf("iter %2i, val %1.5e, err %1.5e, step %1.5e\n",
				num,pVal(r),err,α);
		end
		if num >= itermax
			print("Reach Maximum Iteration!\n");
			break;
		end
	end
	print("τ,κ: ",τ," ",κ,"\n");
end
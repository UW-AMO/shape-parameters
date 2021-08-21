#====================================================================
  Use Quantile as Regularizer to Do Linear Regression
  - define functions
  - train data
  - cross-validation for picking λ
  - test data (report MSE)
====================================================================#
#--------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------
function mse!(r::Array{Float64,1},y::Array{Float64,1},
			  A::Array{Float64,2},x::Array{Float64,1})
	m = length(r);
	BLAS.gemv!('N',1.0,A,x,0.0,r);
	val = 0.0
	for i = 1:m
		r[i] -= y[i];
		val  += r[i]^2;
	end
	return val/m
end
# objective function value
function obj(r::Array{Float64,1},x::Array{Float64,1},
			 τ::Float64,λ::Float64)
	n   = length(x);
	val = 0.0;
	for i = 1:n
		x[i] < 0 ? val += -τ*x[i] : val += (1-τ)*x[i];
	end
	val += -n*log(τ*(1-τ));
	return 0.5*vecnorm(r)^2+λ*val
end
# give x project τ
function proj_τ(x::Array{Float64,1},λ::Float64)
	n = length(x);
	xp = sum(max(x,0));
	xm = sum(max(-x,0));
	a = (xm-xp)/n;
#	a = sum(x)/n;
#	return 2.0/(sqrt(a^2+4.0)+2.0-a)
return 0.5*((1+2/a) +sqrt(1+(2/a)^2))
end
# prox sub problem for calculating τ
function ml_τ(x::Array{Float64,1},λ::Float64)
	n = length(x);
	function f(τ)
		val = 0.0;
		# derivative of quantile-huber w.r.t. τ
		for i = 1:n
			x[i]<   -τ*λ ? val += -x[i]-τ*λ :
			x[i]>(1-τ)*λ ? val += -x[i]+(1-τ)*λ : nothing;
		end
		return val + n*(2.0*τ-1.0)/(τ*(1-τ))
	end
	return fzero(f,0.5)
end
# give τ project x (threshold on x)
function proj_x!(x::Array{Float64,1},τ::Float64,λ::Float64)
	n = length(x);
	for i = 1:n
		x[i]<   -τ*λ ? x[i] +=     τ*λ :
		x[i]>(1-τ)*λ ? x[i] -= (1-τ)*λ : x[i] = 0.0;
	end
end
# gradient of l2 part: A'(Ax-b)
function grad!(g::Array{Float64,1},r::Array{Float64,1},
			   x::Array{Float64,1},y::Array{Float64,1},
			   A::Array{Float64,2})
	m,n = size(A);
	BLAS.gemv!('N',1.0,A,x,0.0,r);
	for i = 1:m r[i] -= y[i]; end
	BLAS.gemv!('T',1.0,A,r,0.0,g);
end
# proximal graident method for regression
function pgdSolver!(x::Array{Float64,1},y::Array{Float64,1},
					A::Array{Float64,2},λ::Float64,τ::Float64,
					tol::Float64,itermax::Int64)
	m,n = size(A);
	r   = zeros(m);
	g   = zeros(n);
	η   = 1.0/norm(A)^2;
	print("step size: ",η,"\n");
	err = 1.0;
	num = 0;
	# gradient of smooth part
	τ = proj_τ(x,λ); # use the lambda
	grad!(g,r,x,y,A);
	while err >= tol
		# forward graident step
		for i = 1:n x[i] -= η*g[i]; end
		# proximal step
		τ = proj_τ(x,λ); # use the lambda
		λ!=0 && proj_x!(x,τ,λ*η);
		# update gradient
		grad!(g,r,x,y,A);
		# err = vecnorm(g);
		err = 0.0;
		for i = 1:n
			x[i] < 0.0 ? err += (g[i]-τ*λ)^2 :
			x[i] > 0.0 ? err += (g[i]+(1-τ)*λ)^2 :
			g[i] > τ*λ ? err += (g[i]-τ*λ)^2 :
			g[i] < (τ-1)*λ ? err += (g[i]+(1-τ)*λ)^2 : nothing;
		end
		num += 1;
		num%10==0 &&
		@printf("iter %7d, obj %1.5e, err %1.5e\n",
			num,obj(r,x,τ,λ),err);
		if num >= itermax
			print("Reach Maximum Iteration!\n");
			break;
		end
	end
	print(num,"\n");
	print("τ = ",τ,"\n");
	# return x
end
#--------------------------------------------------------------------
# Train Data
#--------------------------------------------------------------------
print("--------------------train data--------------------\n");
# load data
print("load data...\n");
fid  = open("data/train.bin","r");
m    = read(fid,Int64)[1];
n    = read(fid,Int64)[1];
A    = read(fid,Float64,m*n);
b    = read(fid,Float64,m);
xt   = read(fid,Float64,n);
close(fid);
A    = reshape(A,m,n);
# train data with different λ
print("start training...\n");
λ    = 0.4;
τ    = 0.5;
x    = zeros(n);
pgdSolver!(x,b,A,λ,τ,1e-6,20000);
print("end training...\n");
#--------------------------------------------------------------------
# Plot Result
#--------------------------------------------------------------------
# using PyPlot
# print("-----------------compare to true x----------------\n");
# err  = norm(x-xt)/norm(xt);
# @printf("relative error to true x: %1.5e\n",err);
# plot(1:n,x,".b",1:n,xt,"-r");
# savefig("qx.eps");
#--------------------------------------------------------------------
# Save Data
#--------------------------------------------------------------------
print("---------------------save data--------------------\n");
fid = open("data/quantilex.bin","w");
write(fid,n,x);
close(fid);
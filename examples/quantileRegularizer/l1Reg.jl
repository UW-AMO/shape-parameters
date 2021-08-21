#====================================================================
  Use l1-Regularizer to Do Linear Regression
  - define functions
  - train data
  - cross-validation for picking λ
  - test data (report MSE) and also see if recovery true x
====================================================================#
#--------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------
# mean square error
function mse!(r::Array{Float64,1},b::Array{Float64,1},
			  A::Array{Float64,2},x::Array{Float64,1})
	m = length(r);
	BLAS.gemv!('N',1.0,A,x,0.0,r);
	val = 0.0
	for i = 1:m
		r[i] -= b[i];
		val  += r[i]^2;
	end
	return val/m
end
# objective function
function obj(r::Array{Float64,1},x::Array{Float64,1},λ::Float64)
	n   = length(x);
	val = 0.5*vecnorm(r)^2;
	val+= λ*sum(abs(x));
	return val
end
# gradient of l2 part: A'(Ax-b)
function grad!(g::Array{Float64,1},r::Array{Float64,1},
			   x::Array{Float64,1},b::Array{Float64,1},
			   A::Array{Float64,2})
	m,n = size(A);
	BLAS.gemv!('N',1.0,A,x,0.0,r);
	for i = 1:m r[i] -= b[i]; end
	BLAS.gemv!('T',1.0,A,r,0.0,g);
end
# prox function w.r.t. l1 penalty
function prox_l1!(x::Array{Float64,1},λ::Float64)
	n = length(x);
	for i = 1:n
		x[i] >  λ ? x[i] -= λ :
		x[i] < -λ ? x[i] += λ : x[i] = 0.0;
	end
end
# proximal gradient solver
function pgdSolver!(x::Array{Float64,1},b::Array{Float64,1},
				   A::Array{Float64,2},λ::Float64,
				   tol::Float64,itermax::Int64)
	m,n = size(A);
	r   = zeros(m);
	g   = zeros(n);
	η   = 1.0/norm(A)^2;
	# print("stepsize: ",η,"\n");
	err = 1.0;
	num = 0;
	# gradient of smooth part
	grad!(g,r,x,b,A);
	while err >= tol
		# forward graident step
		for i = 1:n x[i] -= η*g[i]; end
		# proximal step
		λ != 0.0 && prox_l1!(x,η*λ);
		# update gradient
		grad!(g,r,x,b,A);
		# err = vecnorm(g);
		err = 0.0;
		for i = 1:n
			x[i] < 0.0 ? err += (g[i]-λ)^2 :
			x[i] > 0.0 ? err += (g[i]+λ)^2 :
			g[i] > λ   ? err += (g[i]-λ)^2 :
			g[i] <-λ   ? err += (g[i]+λ)^2 : nothing;
		end
		num += 1;
		num%1000==0 && 
		@printf("iter %5d, obj %1.5e, err %1.5e\n",
			num,obj(r,x,λ),err);
		if num >= itermax
			print("Reach Maximum Iteration!\n");
			break;
		end
	end
	# print(num,"\n");
	return x
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
close(fid);
A   = reshape(A,m,n);
# train data with different λ
print("start training...\n");
# λ = maximum(abs(A'*b))/10;
# λ = linspace(0.41,0.6,20);
λ    = 1.0;
x    = ones(n);
pgdSolver!(x,b,A,λ,1e-6,20000);
print("end training...\n");
#--------------------------------------------------------------------
# Save Data
#--------------------------------------------------------------------
print("---------------------save data--------------------\n");
fid = open("data/l1x.bin","w");
write(fid,n,x);
close(fid);
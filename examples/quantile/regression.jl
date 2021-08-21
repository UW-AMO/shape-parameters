#==========================================================
 Use different objectives to do the regression
 Obj1: Least Square
 Obj2: Quatile (Manully tuned parameter)
 Obj3: Quatile (Self tuned using projective func)
==========================================================#
#----------------------------------------------------------
# Load Data
#----------------------------------------------------------
print("Loading Data...\n");
fid = open("data/data.bin","r");
m   = read(fid,Int64,1)[1];
n   = read(fid,Int64,1)[1];		# size of problem
A   = read(fid,Float64,m*n);	# data matrix
b   = read(fid,Float64,m);		# observed data
xt  = read(fid,Float64,n);		# true x
τt  = read(fid,Float64);		# true τ
close(fid);
A   = reshape(A,(m,n));
# @printf("stepsize: %1.5e\n",norm(A));
#----------------------------------------------------------
# Obj1: Least Square
#----------------------------------------------------------
print("---------------------------------------------\n");
print("Calculating Obj1 Solution...\n");
x1  = A\b;
er1 = norm(x1-xt)/norm(xt);
@printf("least squares Error: %1.5e\n",er1);
#----------------------------------------------------------
# Obj2: Quatile (Manully tuned parameter)
#----------------------------------------------------------
include("./quantileFun.jl");
print("---------------------------------------------\n");
print("Calculating Obj2 Solution...\n");
x2  = ones(n);
tol = 1e-6;
itermax = 500;
# @time(pdSolver!(x2,A,b,τt,tol,itermax));
subBFGS!(x2,A,b,τt,tol,itermax);
er2 = norm(x2-xt)/norm(xt);
@printf(" true quantile Error: %1.5e\n",er2);
#----------------------------------------------------------
# Obj3: Quatile (Self tuned using projection func)
#----------------------------------------------------------
# TODO: Add the projection function (closed form) in while
#	doing regression in subBFGS
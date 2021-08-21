#==========================================================
 Use different objectives to do the regression
 Obj1: Least Square
 Obj2: Quatile-Huber (Manully tuned parameter)
 Obj3: Quatile-Huber (Self tuned parameter)
 Obj4: Quatile-Huber (Self tuned using projective func)
==========================================================#
# using Optim;
#----------------------------------------------------------
# Load Data
#----------------------------------------------------------
print("Loading Data...\n");
fid = open("data/data.bin","r");
m   = read(fid,Int64,1)[1];
n   = read(fid,Int64,1)[1];		# size of problem
A   = read(fid,Float64,m*n);	# data matrix
y   = read(fid,Float64,m);		# observed data
xt  = read(fid,Float64,n);		# true x
τt  = read(fid,Float64);		# true τ
κt  = read(fid,Float64);		# true κ
close(fid);
A   = reshape(A,(m,n));
#----------------------------------------------------------
# Obj1: Least Square
#----------------------------------------------------------
print("---------------------------------------------\n");
print("Calculating Obj1 Solution...\n");
x1  = (A'*A)\(A'*y);
er1 = norm(x1-xt)/norm(xt);
@printf("least squares Error1: %1.5e\n",er1);
#----------------------------------------------------------
# Obj2: Quatile-Huber (Manully tuned parameter)
#----------------------------------------------------------
include("qhFun.jl");
print("---------------------------------------------\n");
print("Calculating Obj2 Solution...\n");
x2  = zeros(n);
tol = 1e-6;
itermax = 5000;
BFGSSolver!(y,A,x2,τt,κt,tol,itermax);
er2 = norm(x2-xt)/norm(xt);
@printf(" true quantile-huber Error: %1.5e\n",er2);
#----------------------------------------------------------
# Obj3: Quatile-Huber (Self tuned parameter)
#----------------------------------------------------------
include("mlFun.jl");
print("---------------------------------------------\n");
print("Calculating Obj3 Solution...\n");
x3  = zeros(n);
τ3  = 0.5;
κ3  = 0.1;
tol = 1e-8;
itermax = 200;
for i = 1:10
	BFGSSolver!(y,A,x3,τ3,κ3,tol,itermax);
	r = A*x3 - y;
	er3 = norm(x3-xt)/norm(xt);
	@printf(" tune quantile-huber Error: %1.5e\n",er3);
	τ3, κ3 = ntSolver(τ3,κ3,r,tol,itermax);
	@printf("τ3, κ3, obj: %1.5e, %1.5e, %1.5e\n",
		τ3,κ3,obj(r,τ3,κ3));
end
#----------------------------------------------------------
# Obj4: Quatile-Huber (Self tuned using projective func)
#----------------------------------------------------------
include("prFun.jl");
print("---------------------------------------------\n");
print("Calculating Obj4 Solution...\n");
x4  = zeros(n);
tol = 1e-8;
itermax = 500;
pSolver!(y,A,x4,tol,itermax);
er4 = norm(x4-xt)/norm(xt);
@printf(" projected quantile-huber Error: %1.5e\n",er4);

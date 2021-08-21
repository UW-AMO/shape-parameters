#==========================================================
 Use different objectives to do the regression
 Obj1: Least Square
 Obj2: Least Square without Outliers
 Obj3: Huber (Manully tuned parameter)
 Obj4: Huber (Self tuned parameter)
 Obj5: Huber (Self tuned using projective func)
==========================================================#
using Optim; # using the LBFGS solver in this package
#----------------------------------------------------------
# Load Data
#----------------------------------------------------------
print("Loading Data...\n");
fid = open("data/data.bin","r");
m   = read(fid,Int64,1)[1];
n   = read(fid,Int64,1)[1];		# size of problem
A   = read(fid,Float64,m*n);	# data matrix
y   = read(fid,Float64,m);		# observed data
yt  = read(fid,Float64,m);		# data without outliers
xt  = read(fid,Float64,n);		# true x
close(fid);
A   = reshape(A,(m,n));
#----------------------------------------------------------
# Obj1: Least Square
#----------------------------------------------------------
print("---------------------------------------------\n");
print("Calculating Obj1 Solution...\n");
x1  = A\y;
er1 = norm(x1-xt)/norm(xt);
@printf("least squares Error1: %1.5e\n",er1);
#----------------------------------------------------------
# Obj2: Least Square without Outliers
#----------------------------------------------------------
print("---------------------------------------------\n");
print("Calculating Obj2 Solution...\n");
x2  = A\yt;
er2 = norm(x2-xt)/norm(xt);
@printf("best least squares Error1: %1.5e\n",er2);
#----------------------------------------------------------
# Obj3: Huber (Manully tuned parameter)
#----------------------------------------------------------
print("---------------------------------------------\n");
print("Calculating Obj3 Solution...\n");
include("./huberFun.jl");
κ   = 4.0;						# decided by observe error
η   = 1.0/norm(A)^2;			# step size
x3  = zeros(n);					# initialize x
tol = 1e-5;						# tolerence
itermax = 5000;					# maximum iteration number
# gdSolver!(y,A,x3,κ,η,tol,itermax);
BFGSSolver!(y,A,x3,κ,tol,itermax);
er3 = norm(x3-xt)/norm(xt);
@printf(" true huber Error: %1.5e\n",er3);
#----------------------------------------------------------
# Obj4: Huber (Self tuned parameter)
#----------------------------------------------------------
print("---------------------------------------------\n");
print("Calculating Obj4 Solution...\n");
include("./mlFun.jl");
κ   = 0.01;
x4  = zeros(n);					# initialize x and κ
tol = 1e-8;
itermax = 20000;
k   = 10;						# alt update number
for i = 1:k
	# update x
	# gdSolver!(y,A,x4,κ,η,tol,itermax);
	BFGSSolver!(y,A,x4,κ,tol,itermax);
	r = A*x4 - y;
	# update κ
	κnew = ntSolver(r,κ,tol,itermax);
	print("update κ = ",κ,"\n");
	er4 = norm(x4-xt)/norm(xt);
	@printf("iter %i, tuning huber Error: %1.5e\n",i,er4);
	if abs(κnew-κ) <= 1e-8
		print("converged!\n");
		break;
	end
	κ = κnew;
end
er4 = norm(x4-xt)/norm(xt);
@printf(" alter-tuning huber Error: %1.5e\n",er4);
#----------------------------------------------------------
# Obj5: Huber (Self tuned with projective func)
#----------------------------------------------------------
print("---------------------------------------------\n");
print("Calculating Obj5 Solution...\n");
include("./prFun.jl");
res = optimize(func,grad!,zeros(n),GradientDescent());
x5  = Optim.minimizer(res);
er5 = norm(x5-xt)/norm(xt);
r = A*x5 - y;
κ = ntSolver(r,0.01,tol,itermax);
# print(κ);
show(res)
@printf(" tuning prohuber Error: %1.5e\n",er5);
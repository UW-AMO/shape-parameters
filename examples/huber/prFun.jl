#====================================================================
 Functions of Huber Objective Projected out κ
 func: function value of huber objective
 grad: gradient of huber objective
====================================================================#
# include("./mlFun.jl"); # necessary funcs for projecting κ
#--------------------------------------------------------------------
# Function Value
#--------------------------------------------------------------------
function func(x::Array{Float64,1})
	# not necessary load data
	# TODO::figure out how to use Optim package efficient
	fid = open("data/data.bin","r");
	m   = read(fid,Int64,1)[1];
	n   = read(fid,Int64,1)[1];		# size of problem
	A   = read(fid,Float64,m*n);	# data matrix
	y   = read(fid,Float64,m);		# observed data
	close(fid);
	A   = reshape(A,(m,n));
	# calculate optimal κ
	r = A*x - y;
	tol = 1e-10;
	itermax = 100;
	κ = ntSolver(r,0.01,tol,itermax);
	# print("κ:",κ,"\n");
	# value of huber function
	val = 0.0;
	for i = 1:m
		if abs(r[i]) <= κ
			val += 0.5*r[i]^2;
		else
			val += 0.5*abs(r[i]) - 0.5*κ^2;
		end
	end
	return val
end
#--------------------------------------------------------------------
# Gradient
#--------------------------------------------------------------------
function grad!(x::Array{Float64,1},g::Array{Float64,1})
	# not necessary load data
	# TODO::figure out how to use Optim package efficient
	fid = open("data/data.bin","r");
	m   = read(fid,Int64,1)[1];
	n   = read(fid,Int64,1)[1];		# size of problem
	A   = read(fid,Float64,m*n);	# data matrix
	y   = read(fid,Float64,m);		# observed data
	close(fid);
	A   = reshape(A,(m,n));
	# calculate optimal κ
	r = A*x - y;
	tol = 1e-10;
	itermax = 100;
	κ = ntSolver(r,0.01,tol,itermax);
	# gradient of huber function
	for i = 1:m
		r[i]> κ?r[i]= κ:
		r[i]<-κ?r[i]=-κ:0;
	end
	r = A'*r;
	# this is wired
	for i = 1:n g[i] = r[i]; end
	# print("g:",norm(r),"\n");
end
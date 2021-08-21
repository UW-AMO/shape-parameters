#====================================================================
  Used for Generating Data of Regularizer Experiment
  m: number of rows of A
  n: number of columns of A
  A: given matrix
  b: observed data
  x: the vector we trying to fit
     sparse and with certian percentage of sign
====================================================================#
# using PyPlot;
srand(123)
m = 200;
n = 512;
α = 1.0/sqrt(n);
# using qr decomposition to get better condition number
# A = α*randn(m,n);
A,_ = qr(randn(n,m));
A = A';
x = zeros(n);
β = 0.02;
# k = round(Int64,β*n);				# sparsitb of x
k = 20;
p = randperm(n)[1:k];
τ = 1.0;                    # ratio of + sign against - sign
for i = 1:k
	if rand() <= τ
		s =  1.0; μ = 10.0;			# magnitude of positive spike
	else
		s = -1.0; μ = 10.0;			# magnitude of negative spike
	end
	x[p[i]] = s*(abs(randn())+μ);
end

# generate observation data b
b = A*x - 0.1*randn(m);
# save data
fid = open("data/train.bin","w");
write(fid,m,n);
write(fid,A,b);
write(fid,x);
close(fid);
# true x
fid = open("data/true_x.bin","w");
write(fid,n,x);
close(fid);
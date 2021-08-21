#==========================================================
 Used for generating simulation data
 m: number of rows of A
 n: number of columns of A
 A: given matrix
 y: observed data
 x: the vector we trying to fit
 err: errors comes from a certain distribution
==========================================================#
include("sampleQ.jl");
srand(123);
m   = 2000;
n   = 50;				# size of the problem
α   = 100;				# scaling of the problem
A   = α*randn(m,n);
x   = randn(n);
# sample errors from the quantile-huber distribution
τ   = 0.9;
err = zeros(m);
for i = 1:m
	err[i] = sampleQ(τ);
end
y   = A*x - err;
# save the data
fid = open("data/data.bin","w");
write(fid,m);
write(fid,n);
write(fid,A);
write(fid,y);
write(fid,x);
write(fid,τ);
close(fid);

fid = open("data/err.bin","w");
write(fid,m);
write(fid,err);
close(fid);
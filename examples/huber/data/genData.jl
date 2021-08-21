#==========================================================
 Used for generating simulation data
 m: number of rows of A
 n: number of columns of A
 A: given matrix
 y: observed data
 x: the vector we trying to fit
 err: mix errors we add to y
==========================================================#
srand(123);
m   = 2000;
n   = 50;				# size of the problem
α   = 100;				# scaling of the problem
r   = 0.5;				# outlier ratio
A   = α*randn(m,n);
x   = randn(n);
# add guassian nosie
err = randn(m);
# add outliers
k   = ceil(Integer,m*r);
ind = randperm(m)[1:k];
ero = rand(k);
# ero = rand(k)-0.5;
# ero = maximum(abs(err))*5*ero/minimum(abs(ero));
yt  = A*x + err;
out = maximum(abs(err))*100.;
for i = 1:k
	s = rand([-1.,1.]);
	err[ind[i]] += s*(out + ero[i]);
end
y   = A*x + err;
# save the data
fid = open("data/data.bin","w");
write(fid,m);
write(fid,n);
write(fid,A);
write(fid,y);
write(fid,yt);
write(fid,x);
close(fid);

fid = open("data/err.bin","w");
write(fid,m);
write(fid,err);
close(fid);
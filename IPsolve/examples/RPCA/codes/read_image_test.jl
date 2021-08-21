#======= Read Image Files =======#
using PyPlot

# load data matrix
m,n = 20480, 1439;  # size of matrix
fid = open("Campus/trees.bin","r");
A   = read(fid, UInt8, m*n);
A   = reshape(A,m,n);
close(fid);

# check recover original picture
mp, np = 128,160;   # size of one image
# imshow(reshape(A[:,1],mp,np), cmap="gray");
# savefig("sample.eps");

# convert to [0,1] scale
A = A/255;
imshow(reshape(A[:,1],mp,np), cmap="gray");
savefig("sample.eps");

# save the float number data matrix
fid = open("Campus/trees_float.bin","w");
write(fid, A);
close(fid);
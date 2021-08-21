#======= Test the Result Obtained from RPCA =======#
using PyPlot

m,n = 20480, 202;  # size of matrix
mp, np = 128,160;   # size of one image
# load data
fid = open("../results/LS.bin","r");
L   = read(fid,Float64,m*n);
S   = read(fid,Float64,m*n);
close(fid);
L   = reshape(L,m,n);
S   = reshape(S,m,n);

# plot result
# id = [1209,1362,1664,1702,1816,2017,2345,2362,2384]-999;
# id = [1209,1362,1664]-999;
# id = [1209] - 999;
id = [202]

for I in eachindex(L)
    L[I] < 0.0 ? L[I] = 0.0 :
    L[I] > 1.0 ? L[I] = 1.0 : continue
end
for I in eachindex(S)
    S[I] < 0.0 ? S[I] = 0.0 :
    S[I] > 1.0 ? S[I] = 1.0 : continue
end
@show(maximum(S),minimum(S))
for i in id
    imshow([reshape(L[:,i],mp,np);reshape(S[:,i],mp,np)], cmap="gist_gray");
    axis("off")
    savefig(@sprintf("../pics/LS%d.eps",i+999),bbox_inches="tight");
    # imshow(reshape(S[:,i],mp,np), cmap="gray");
    # axis("off")
    # savefig(@sprintf("../pics/S%d.eps",i+999),bbox_inches="tight");
end

# m,n = 20480, 1439;  # size of matrix
# fid = open("../Campus/trees_float.bin","r");
# A   = read(fid, Float64, m*n);
# A   = reshape(A,m,n);
# close(fid);
# A   = A[:,1385]; # sub set
# imshow(reshape(A,mp,np), cmap="gray");
# axis("off")
# savefig(@sprintf("../pics/Y%d.eps",202+999),bbox_inches="tight");
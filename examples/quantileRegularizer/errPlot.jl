using PyPlot
# load data
fid = open("data/l1x.bin","r");
n   = read(fid,Int64,1)[1];
x1  = read(fid,Float64,n);
close(fid);
fid = open("data/quantilex.bin","r");
n   = read(fid,Int64,1)[1];
xq  = read(fid,Float64,n);
close(fid);
fid = open("data/true_x.bin","r");
n   = read(fid,Int64,1)[1];
xt  = read(fid,Float64,n);
close(fid);

plot(1:n,xt,"-r",1:n,x1,".b",1:n,xq,".g");
legend(["true x","l1-Reg","quantile-Reg"],fontsize=5.0,loc="upper left");
title("compare result");
xlabel(L"$x$");
savefig("compareAll.eps");
#====================================================================
    RPCA Experiments
    Data:
        http://vis-www.cs.umass.edu/~narayana/castanza/I2Rdataset/
        Campus.zip

====================================================================#

##### load packages and functions #####
include("./RPCA_SolverT.jl")

##### load data #####
println("load data...")
m,n = 20480, 1439;  # size of matrix
fid = open("../Campus/trees_float.bin","r");
A   = read(fid, Float64, m*n);
A   = reshape(A,m,n);
close(fid);
A   = A[:,[collect(1:200);1384;1385]]; # sub set


##### set up parameters #####
k   = 2;
κ   = 5.5;
σ   = 0.025;
λ   = 0.0;
# svd initialization
println("svd initialization...")
UΣV,nconv,niter,nmult,resid = svds(A,nsv=k,tol=1e-12)
U   = diagm(sqrt(UΣV[:S]))*UΣV[:U].';
V   = diagm(sqrt(UΣV[:S]))*UΣV[:V];
for I in eachindex(U)
    U[I] < 0.0 ? U[I] = 0.0 :
    U[I] > 1.0 ? U[I] = 1.0 : continue
end
for I in eachindex(V)
    V[I] < 0.0 ? V[I] = 0.0 :
    V[I] > 1.0 ? V[I] = 1.0 : continue
end
# U = rand(k,m);
# V = rand(k,202);
# solver parameters
params = solver_params(1e-12,300,1);

##### apply solver #####
println("start solver...")
θ = RPCA_Solver!(U,V,[κ,σ],A,λ,params,fixθ=false);
# κ   = RPCA_Solver!(U,V,κ,A,λ,params,fixκ=true);
# κ,σ = θ[1],θ[2]
##### recover S and save data #####
m,n = size(A);
println("recover S...")
L = U.'*V;
r = A - L;
κσ = θ[1]*θ[2];
# κσ = κ;
S = zeros(m,n);
for I in eachindex(S)
    r[I] >  κσ ? S[I] = r[I] - κσ :
    r[I] < -κσ ? S[I] = r[I] + κσ : continue;
end


println("save data...")
fid = open("../results/LS.bin","w");
write(fid,L,S);
close(fid);
fid = open("../results/R.bin","w");
write(fid,r);
close(fid);
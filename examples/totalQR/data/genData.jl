#====================================================================
  Generating Data for Total Quatile Regression
====================================================================#
include("./sampleQ.jl")
function genData(m::Int64, n::Int64; sd = false)
    srand(123)
    At  = rand(m,n);        # true collected feature data
    τt  = rand(n);          # τ values for feature errs distribution
    # τt  = fill(0.2,n);
    # τt  = [0.7,0.1,0.1];
    A   = zeros(m,n);       # error matrix
    α   = 1.0;              # quantile error magnitutde
    β   = 1.0;              # gaussian error magnitutde
    # sample the quantile noise and create contaminated data matrix
    for j = 1:n, i = 1:m
        A[i,j] = At[i,j] + α*sampleQ(τt[j]);
    end
    xt  = 10*rand(n);         # true feature coefficient
    err = randn(m);
    y   = At*xt + β*err;
    if !sd
        # save the data
        fid = open("data/data.bin","w");
        write(fid,m,n);
        write(fid,A,y);
        write(fid,At,xt,τt);     # all the true data
        close(fid);
    end
    return A, y, At, xt, τt
end
#======= Test NLopt Package for projection of θ =======#
using NLopt

##### Load Data #####
m,n = 20480, 202;
fid = open("../results/R.bin","r")
R   = read(fid,Float64,m*n)
close(fid)

##### Define Functions #####
count = 0
function θfun(θ::Vector, g::Vector, R::Vector)
    n   = length(R);
    κ,σ = θ[1],θ[2];
    val = 0.0;
    # gradient and hessian of huber penalty w.r.t. κ
    for I in eachindex(R)
        if abs(R[I]) > κ*σ
            # gradient
            g[1] += abs(R[I])/σ - κ;
            g[2] -= κ*abs(R[I])/σ^2;
            val  += κ*abs(R[I])/σ-0.5*κ^2;
        else
            # gradient
            g[2] -= R[I]^2/σ^3;
            val  += 0.5*R[I]^2/σ^2;
        end
    end
    val /= n;
    scale!(g,1/n);
    # gradient and hessian of log normalization constant
    eκ  = exp(-0.5*κ^2);
    c   = 2.0*eκ/κ + sqrt(2.0*pi)*erf(κ/sqrt(2.0));
    dc  = -2.0*eκ/κ^2;
    ddc = (4.0/κ^3+2.0/κ)*eκ;
    g[1] += dc/c;
    g[2] += 1.0/σ;
    val  += log(c) + log(σ);

    global count
    count::Int += 1
    println("f_$count($θ) = $val")

    return val
end

function θfun(θ::Vector, R::Vector)
    n   = length(R);
    κ,σ = θ[1],θ[2];
    val = 0.0;
    # gradient and hessian of huber penalty w.r.t. κ
    for I in eachindex(R)
        if abs(R[I]) > κ*σ
            val  += κ*abs(R[I])/σ-0.5*κ^2;
        else
            val  += 0.5*R[I]^2/σ^2;
        end
    end
    val /= n;
    # gradient and hessian of log normalization constant
    c   = 2.0*eκ/κ + sqrt(2.0*pi)*erf(κ/sqrt(2.0));
    val  += log(c) + log(σ);

    global count
    count::Int += 1
    println("f_$count($θ) = $val")

    return val
end

myfunc = (θ::Vector,g::Vector) -> θfun(θ,g,R)
# myfunc = θ -> θfun(θ,R)

##### Call Solver #####
opt = Opt(:LD_MMA, 2)
lower_bounds!(opt,[1e-4,1e-4])
xtol_rel!(opt,1e-6)

min_objective!(opt,myfunc)
(minf,minx,ret) = optimize(opt,[1e-2,1e-2])
println("got $minf at $minx (returned $ret)")
@show minx[1]/minx[2]
@show m*n/sum(abs(R))
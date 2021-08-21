#====================================================================
    Random Search Find Global Minimum Inside Box
    minₓ f(x)   s.t. x ∈ [a,b]

    Input:
        - f: function handle
        - a: left end point
        - b: right end point
        - x: initial point
    Output:
        - x: minimizer
        - fval: optimal function value
====================================================================#

function rminbnd(f, a, b, x; tol=1e-6)
    # parameters for random search
    p = 30; # populartion number
    s = 5;  # number of seeds
    k = div(p-s,s);    # number of random mutations per seed
    m = round(Int64,log(tol)/log((b-a)/p))+1; # number of generation
    σ = sqrt(pi/2); # scale for mutations
    # initial generation
    spa = (b-a)/(p-2);
    gen = zeros(p);  # uniform distributed
    val = zeros(p);
    ind = zeros(Int64,p);
    gen[1] = x; val[1] = f(gen[1]);
    for i = 2:p
        gen[i] = a + (i-2)*spa;
        val[i] = f(gen[i]);
    end
    seed = zeros(s);
    # start search
    for i = 1:m
        sortperm!(ind,val);
        for j = 1:s seed[j] = gen[ind[j]]; end
        for j = 1:s
            sid = 1 + (j-1)*(k+1)
            gen[sid] = seed[j]
            r   = σ/p^j
            for l = 1:k
                gen[sid+l] = seed[j] + r*randn();
                gen[sid+l] < a ? gen[sid+l] = a :
                gen[sid+l] > b ? gen[sid+l] = b : continue;
            end
        end
        for j = 1:p val[j] = f(gen[j]); end
    end
    return gen[1], val[1]
end
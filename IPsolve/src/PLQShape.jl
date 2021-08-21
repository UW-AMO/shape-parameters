#= ===================================================================
    Module for PLQ Shape Parameters Fitting

    Consider problem,
        (𝒫) min_{x,Sᵀθ ≤ s} ρ(A⋅x - y,θ) + m⋅log[c(θ)]
    where c(θ) = ∫ exp(-ρ(r,θ)) dr.

    Dual representation of ρ(A⋅x-y,θ),
        (𝒟) sup_{Cᵀu ≤ Hᵀθ + c} {uᵀ[B(A⋅x-y) - Gᵀθ - b] - uᵀ⋅M⋅u/2}
=================================================================== =#

module PLQShape

export PLQ_Primal, PLQ_Dual, IPsolve, IP_params, PALMProj, PALMProj_params, projθ,
        PALM, PALM_params

##### data structure for PLQ #####
mutable struct PLQ_Primal
    # given data
    A::Array{Float64,2}
    y::Array{Float64,1}
    # primal variables
    x::Array{Float64,1}
    θ::Array{Float64,1}
    # primal constraints
    S::Array{Float64,2}
    s::Array{Float64,1}
    # primal objective
    ρ::Function         # PLQ penalty
    L₁::Function        # Lipschitz constant w.r.t. x when fix θ
    lognc::Function     # log normalization constant
end

mutable struct PLQ_Dual
    # parameters in the dual representation of PLQ penalty
    B::Array{Float64,2}
    G::Array{Float64,2}
    b::Array{Float64,1}
    M::Array{Float64,2}
    # constraints
    C::Array{Float64,2}
    H::Array{Float64,2}
    c::Array{Float64,1}
end

##### algorithms #####
include("Algorithms/IPsolve.jl")
include("Algorithms/PALMProj.jl")
include("Algorithms/PALM.jl")

end
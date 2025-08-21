using Test
using TMLE
using Random
using Distributions
using LinearAlgebra
using DataFrames
using ToeplitzMatrices

function simulate_highdim_lasso_data(n::Int=1000, p::Int=100, rho::Float64=0.9, k::Int=20, amplitude::Float64=1, amplitude2::Float64=1, k2::Int=20)
    function toeplitz_cov(p, rho)
        v = rho .^ (0:(p-1))
        return Matrix(Toeplitz(v,v))
    end

    Sigma = toeplitz_cov(p, rho)
    mu = zeros(Float64, p)
    W = (rand(MvNormal(mu, Sigma), n))'
    W = (W .- mean(W, dims=1)) ./ std(W, dims=1)

    nonzero = sample(1:p, k, replace=false)
    sign = sample([-1, 1], p, replace=true)
    gamma_ = amplitude .* sign .* in(1:p, nonzero)

    nonzero2 = sample(1:p, k2, replace=false)
    sign2 = sample([-1, 1], p, replace=true)
    beta_ = amplitude2 .* sign2 .* in(1:p, nonzero2)
    logit_p = W * beta_
    prob_A = 1 ./ (1 .+ exp.(-logit_p))
    A = [rand(Bernoulli(p)) for p in prob_A]

    Y = 2 .* A .+ W * gamma_ .+ randn(n)

    colnames = [string("W", i) for i in 1:p]
    append!(colnames, ["A", "Y"])
    data = DataFrame(hcat(W, A, Y), colnames)
    return data
end

@testset "LassoCTMLE on simulated data" begin
    Random.seed!(2024)  
    dataset = simulate_highdim_lasso_data(500, 10, 0.3, 3, 2.0, 2.0, 3) 
    confounders = Symbol.([string("W", i) for i in 1:10])
    Ψ = ATE(
        outcome = :Y,
        treatment_values = (A = (case = 1, control = 0),),
        treatment_confounders = (A = confounders,)
    )
    lasso_estimator = Tmle(
        collaborative_strategy = LassoCTMLE(
            confounders = confounders
        )
    )
    lasso_result, _ = lasso_estimator(Ψ, dataset; verbosity = 0)
    @test !isnan(estimate(lasso_result))
    @info "LassoCTMLE estimate on simulated data:" estimate(lasso_result)
end
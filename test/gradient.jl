module TestGradient

using TMLE
using Test
using MLJBase
using StableRNGs
using LogExpFunctions
using Distributions
using MLJLinearModels
using MLJModels

μY(T, W)  = 1 .+ 2T .- W.*T

function one_treatment_dataset(;n=100)
    rng = StableRNG(123)
    W   = rand(rng, n)
    μT  = logistic.(1 .- 3W)
    T   = rand(rng, n) .< μT
    Y   = μY(T, W) .+ rand(rng, Normal(0, 0.1), n)
    return (
        T = categorical(T, ordered=true),
        Y = Y,
        W = W
    )
end

@testset "Test gradient_and_estimate" begin
    ps_lowerbound = 1e-8
    Ψ = ATE(
        outcome         = :Y, 
        treatment       = (T=(case=1, control=0),), 
        confounders     = [:W], 
        treatment_model = LogisticClassifier(),
        outcome_model   = with_encoder(InteractionTransformer(order=2) |> LinearRegressor())
    )
    dataset = one_treatment_dataset(;n=100)
    fit!(Ψ, dataset, verbosity=0)
    
    # Retrieve outcome model
    Q = TMLE.getQ(Ψ)
    linear_model = fitted_params(Q).deterministic_pipeline.linear_regressor
    intercept = linear_model.intercept
    coefs = Dict(linear_model.coefs)
    # Counterfactual aggregate
    ctf_agg = TMLE.counterfactual_aggregate(Ψ, Q)
    expected_ctf_agg = (intercept .+ coefs[:T] .+ dataset.W.*coefs[:W] .+ dataset.W.*coefs[:T_W]) .- (intercept .+ dataset.W.*coefs[:W])
    @test ctf_agg ≈ expected_ctf_agg atol=1e-10
    # Retrieve propensity score model
    G = Ψ.scm[:T].mach
    # Gradient Y|X
    H = 1 ./ pdf.(predict(G), dataset.T) .* [t == 1 ? 1. : -1. for t in dataset.T]
    expected_∇YX = H .* (dataset.Y .- predict(Q))
    ∇YX = TMLE.∇YX(Ψ, Q; ps_lowerbound=ps_lowerbound)
    @test expected_∇YX == ∇YX
    # Gradient W
    expectedΨ̂ = mean(ctf_agg)
    ∇W = TMLE.∇W(ctf_agg, expectedΨ̂)
    @test ∇W == ctf_agg .- expectedΨ̂
    # gradient_and_estimate
    IC, Ψ̂ = TMLE.gradient_and_estimate(Ψ, Q; ps_lowerbound=ps_lowerbound)
    @test expectedΨ̂ == Ψ̂
    @test IC == ∇YX .+ ∇W
end

end

true
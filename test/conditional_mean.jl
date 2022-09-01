module ConditionalMean

using Test
using StableRNGs
using TMLE
using DataFrames
using Distributions
using MLJLinearModels
using CategoricalArrays

function build_dataset()
    rng = StableRNG(123)
    n = 100
    # Confounders
    W₁ = rand(rng, n)
    W₂ = rand(rng, n)
    # Covariates
    C₁ = rand(rng, n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< TMLE.expit(0.5sin.(W₁) .- 1.5W₂)
    T₂ = rand(rng, Uniform(), n) .< TMLE.expit(-3W₁ - 1.5W₂)
    # target | Confounders, Covariates, Treatments
    y = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .+ T₁ + rand(rng, Normal(0, 0.1), n)
    return DataFrame(
        T₁= categorical(T₁),
        W₁ = W₁, 
        W₂ = W₂,
        C₁=C₁,
        y=y
        )
end

@testset "Test ATE estimation" begin
    dataset = build_dataset()
    # Define the parameter of interest
    Ψ = ATE(
        target=:y,
        treatment=(T₁=(case=1, control=0),),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    # Define the nuisance parameters specification
    η_spec = (
        Q = LinearRegressor(),
        G = LogisticClassifier()
    )
    # Run TMLE
    result, resultᵢ, cache = tmle(Ψ, η_spec, dataset; verbosity=1, threshold=1e-8);

end

end

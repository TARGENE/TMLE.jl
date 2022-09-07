module Test3pointsInteractions

include("interaction_transformer.jl")

using Random
using StableRNGs
using Distributions
using TMLE
using MLJLinearModels
using CategoricalArrays

interacter = InteractionTransformer(order=3) |> LinearRegressor()

function make_dataset(;n=1000)
    rng = StableRNG(123)
    W = rand(rng, Uniform(), n)
    T₁ = rand(rng, Uniform(), n) .< W
    T₂ = rand(rng, Uniform(), n) .< TMLE.expit(1 .- 2W)
    T₃ = rand(rng, [0, 1], n)

    y = 2 .- 2T₁.*T₂.*T₃.*W + rand(rng, Normal(0, 0.03), n)
    return (W=W, T₁=categorical(T₁), T₂=categorical(T₂), T₃=categorical(T₃), y=y)
end

dataset = make_dataset(;n=1000)
Ψ = IATE(
    target      = :y,
    confounders = [:W],
    treatment   = (T₁=(case=1, control=0), T₂=(case=1, control=0), T₃=(case=1, control=0))
)
η_spec = NuisanceSpec(
    LogisticClassifier(lambda=0),
    interacter
)
tmle_result, initial_result, cache = tmle(Ψ, η_spec, dataset)
mean(tmle_result.IC)
mean(initial_result.IC)
OneSampleTTest(tmle_result)
OneSampleTTest(initial_result)
end

true
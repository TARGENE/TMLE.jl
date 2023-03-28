module Test3pointsInteractions

using Random
using StableRNGs
using Distributions
using TMLE
using MLJLinearModels
using MLJModels
using CategoricalArrays
using Test
using LogExpFunctions

include("helper_fns.jl")

interacter = InteractionTransformer(order=3) |> LinearRegressor()

function make_dataset(;n=1000)
    rng = StableRNG(123)
    W = rand(rng, Uniform(), n)
    T₁ = rand(rng, Uniform(), n) .< W
    T₂ = rand(rng, Uniform(), n) .< logistic.(1 .- 2W)
    T₃ = rand(rng, [0, 1], n)

    y = 2 .- 2T₁.*T₂.*T₃.*(W .+ 10) + rand(rng, Normal(0, 0.03), n)
    return (W=W, T₁=categorical(T₁), T₂=categorical(T₂), T₃=categorical(T₃), y=y), -21
end

@testset "Test 3-points interactions" begin
    dataset, Ψ₀ = make_dataset(;n=1000)
    Ψ = IATE(
        target      = :y,
        confounders = [:W],
        treatment   = (T₁=(case=true, control=false), T₂=(case=true, control=false), T₃=(case=1, control=0))
    )
    η_spec = NuisanceSpec(
        interacter,
        LogisticClassifier(lambda=0),
    )

    tmle_result, cache = tmle(Ψ, η_spec, dataset, verbosity=0)
    # TMLE 
    lb, ub = confint(OneSampleTTest(tmle_result.tmle))
    @test lb ≤ Ψ₀ ≤ ub
    # OneStep
    lb, ub = confint(OneSampleTTest(tmle_result.onestep))
    @test lb ≤ Ψ₀ ≤ ub
    # Risk decreased by fluctuation
    test_fluct_decreases_risk(cache, target_name=:y)

end

end

true
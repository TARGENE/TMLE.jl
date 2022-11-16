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
        treatment   = (T₁=(case=1, control=0), T₂=(case=1, control=0), T₃=(case=1, control=0))
    )
    η_spec = NuisanceSpec(
        interacter,
        LogisticClassifier(lambda=0),
    )

    tmle_result, initial_result, cache = tmle(Ψ, η_spec, dataset, verbosity=0)
    Ψ̂ = TMLE.estimate(tmle_result)
    lb, ub = confint(OneSampleTTest(tmle_result))
    @test lb ≤ Ψ̂ ≤ ub
    @test Ψ̂ ≈ -20.989 atol=1e-3
    # The initial estimate also has the correct answer
    @test TMLE.estimate(initial_result) ≈ -21.008 atol=1e-3
end

end

true
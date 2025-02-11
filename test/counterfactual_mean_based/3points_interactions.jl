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

include(joinpath(dirname(dirname(pathof(TMLE))), "test", "helper_fns.jl"))

function dataset_scm_and_truth(;n=1000)
    rng = StableRNG(123)
    W = rand(rng, Uniform(), n)
    T₁ = rand(rng, Uniform(), n) .< W
    T₂ = rand(rng, Uniform(), n) .< logistic.(1 .- 2W)
    T₃ = rand(rng, [0, 1], n)

    Y = 2 .- 2T₁.*T₂.*T₃.*(W .+ 10) + rand(rng, Normal(0, 0.03), n)
    dataset = (W=W, T₁=categorical(T₁), T₂=categorical(T₂), T₃=categorical(T₃), Y=Y)
    Ψ₀ = -21
    return dataset, Ψ₀
end

@testset "Test 3-points interactions" begin
    dataset, Ψ₀ = dataset_scm_and_truth(;n=1000)
    Ψ = AIE(
        outcome   = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false), 
            T₃=(case=1, control=0)
        ),
        treatment_confounders = (T₁=[:W], T₂=[:W], T₃=[:W])
    )
    models = Dict(
        :Y  => with_encoder(InteractionTransformer(order=3) |> LinearRegressor()),
        :T₁ => LogisticClassifier(lambda=0),
        :T₂ => LogisticClassifier(lambda=0),
        :T₃ => LogisticClassifier(lambda=0)
    )

    tmle = TMLEE(models=models, machine_cache=true, max_iter=3, tol=0)
    result, cache = tmle(Ψ, dataset, verbosity=0);
    test_coverage(result, Ψ₀)
    test_fluct_decreases_risk(cache)
    test_mean_inf_curve_almost_zero(result; atol=1e-10)

    tmle.weighted = true
    result, cache = tmle(Ψ, dataset, verbosity=0)
    test_coverage(result, Ψ₀)
    test_mean_inf_curve_almost_zero(result; atol=1e-10)

    # CHecking cache accessors
    @test length(gradients(cache)) == 3
    @test length(estimates(cache)) == 3
    @test length(epsilons(cache)) == 3
    @test report(cache) isa NamedTuple

end 

end

true
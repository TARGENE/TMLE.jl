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

include(joinpath(dirname(@__DIR__), "helper_fns.jl"))

function dataset_scm_and_truth(;n=1000)
    rng = StableRNG(123)
    W = rand(rng, Uniform(), n)
    T₁ = rand(rng, Uniform(), n) .< W
    T₂ = rand(rng, Uniform(), n) .< logistic.(1 .- 2W)
    T₃ = rand(rng, [0, 1], n)

    Y = 2 .- 2T₁.*T₂.*T₃.*(W .+ 10) + rand(rng, Normal(0, 0.03), n)
    dataset = (W=W, T₁=categorical(T₁), T₂=categorical(T₂), T₃=categorical(T₃), Y=Y)
    truth = -21
    return dataset, truth
end

@testset "Test 3-points interactions" begin
    dataset, Ψ₀ = dataset_scm_and_truth(;n=1000)
    Ψ = IATE(
        outcome   = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false), 
            T₃=(case=1, control=0)
        ),
        treatment_confounders = (T₁=[:W], T₂=[:W], T₃=[:W])
    )
    models = (
        Y = TreatmentTransformer() |> InteractionTransformer(order=3) |> LinearRegressor(),
        T₁ = LogisticClassifier(lambda=0),
        T₂ = LogisticClassifier(lambda=0),
        T₃ = LogisticClassifier(lambda=0)
    )

    tmle = TMLEE(models)
    result, cache = tmle(Ψ, dataset, verbosity=0);
    test_coverage(result, Ψ₀)
    test_fluct_decreases_risk(cache)
    test_mean_inf_curve_almost_zero(result; atol=1e-10)

    tmle.weighted = true
    result, cache = tmle(Ψ, dataset, verbosity=0)
    test_coverage(result, Ψ₀)
    test_mean_inf_curve_almost_zero(result; atol=1e-10)

end 

end

true
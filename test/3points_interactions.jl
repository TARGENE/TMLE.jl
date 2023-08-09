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

function dataset_scm_and_truth(;n=1000)
    rng = StableRNG(123)
    W = rand(rng, Uniform(), n)
    T₁ = rand(rng, Uniform(), n) .< W
    T₂ = rand(rng, Uniform(), n) .< logistic.(1 .- 2W)
    T₃ = rand(rng, [0, 1], n)

    Y = 2 .- 2T₁.*T₂.*T₃.*(W .+ 10) + rand(rng, Normal(0, 0.03), n)
    dataset = (W=W, T₁=categorical(T₁), T₂=categorical(T₂), T₃=categorical(T₃), Y=Y)
    scm = StaticConfoundedModel(
        [:Y], 
        [:T₁, :T₂, :T₃], 
        :W,
        outcome_model = TreatmentTransformer() |> InteractionTransformer(order=3) |> LinearRegressor(),
        treatment_model = LogisticClassifier(lambda=0)
        )
    truth = -21
    return dataset, scm, truth
end

@testset "Test 3-points interactions" begin
    dataset, scm, Ψ₀ = dataset_scm_and_truth(;n=1000)
    Ψ = IATE(
        scm       = scm,
        outcome   = :Y,
        treatment = (T₁=(case=true, control=false), T₂=(case=true, control=false), T₃=(case=1, control=0))
    )

    result, fluctuation = tmle!(Ψ, dataset, verbosity=0)
    test_coverage(result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation)
    test_mean_inf_curve_almost_zero(result; atol=1e-10)

    result, fluctuation = tmle!(Ψ, dataset, verbosity=0, weighted_fluctuation=true)
    test_coverage(result, Ψ₀)
    test_mean_inf_curve_almost_zero(result; atol=1e-10)

end 

end

true
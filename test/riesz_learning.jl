module TestRieszNet

using Test
using TMLE
using RieszLearning
using Random
using StableRNGs
using Distributions
using LogExpFunctions
using CategoricalArrays
using DataFrames
using MLJModels

TEST_DIR = joinpath(pkgdir(TMLE), "test")

include(joinpath(TEST_DIR, "counterfactual_mean_based", "ate_simulations.jl"))

riesznet = RieszNetModel(
    lux_model=RieszLearning.MLP([5, 32, 8]),
    hyper_parameters=(
        rng=Xoshiro(123),
        batch_size=10,
        nepochs=100
    )
)

Ψ = ATE(
    outcome = :Y,
    treatment_values = (T=(case=1, control=0),),
    treatment_confounders = (T = [:W₁, :W₂, :W₃],)
)

@testset "Test RieszRepresenterEstimator" begin
    # Define atch of estimators
    # using the RieszNetModel as the representer
    models = default_models(Q_continuous=DeterministicConstantRegressor())
    models[:riesz_representer] = riesznet
    resampling = CV()
    riesz_estimators = (
        tmle = Tmle(models=models),
        ose = Ose(models=models),
        cv_tmle = Tmle(models=models, resampling=resampling),
        cv_ose = Ose(models=models, resampling=resampling)
    )

    tmle = riesz_estimators.tmle
    ns = [100, 1000, 10_000, 100_000]
    results = []
    for n in ns
        dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=n)
        Ψ̂_tmle, cache = tmle(Ψ, dataset, verbosity=0);
        Ψ̂_tmle
        push!(results, Ψ̂_tmle)
    end
    TMLE.estimate.(results)


    riesznet = RieszNetModel(
        lux_model=RieszLearning.MLP([5, 10]),
        hyper_parameters=(
            rng=Xoshiro(123),
            batch_size=10,
            nepochs=5
        )
    )
    r = TMLE.RieszRepresenter(Ψ)

    X, y = TMLE.get_mlj_inputs_and_target(r, dataset)

    mach = machine(riesznet, X, y)
    fit!(mach, verbosity=1)

    ose = Ose(models=models)
    Ψ̂_ose, _ = ose(Ψ, dataset)
end

end

true
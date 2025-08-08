module TestRieszNet

using Pkg
Pkg.add(url="https://github.com/olivierlabayle/RieszLearning.jl", rev="main") # This is not ideal but adding this to the Manifest leads to issues in CI since the package is not registered.
using Test
using TMLE
using RieszLearning
using Random
using StableRNGs
using Distributions
using LogExpFunctions
using CategoricalArrays
using DataFrames
using MLJBase
using Suppressor
import MLJModels
using Logging

TEST_DIR = joinpath(pkgdir(TMLE), "test")

include(joinpath(TEST_DIR, "counterfactual_mean_based", "ate_simulations.jl"))

function filter_warnings(logger)
    return filter(!=(nothing), map(logger.logs) do log
        if log.level == Logging.Info
            log.message
        end
    end)
end

Ψ = ATE(
    outcome = :Y,
    treatment_values = (T=(case=1, control=0),),
    treatment_confounders = (T = [:W₁, :W₂, :W₃],)
)

riesz_learner(;hidden_layer=10, nepochs=5, batch_size=6) = RieszLearning.RieszNetModel(
    lux_model=RieszLearning.MLP([5, hidden_layer]),
    hyper_parameters=(
        batch_size=batch_size,
        nepochs=nepochs,
        rng=StableRNG(123),
    )
)

@testset "Test build_treatments_factor_estimator" begin
    # When the estimand in the RieszRepresenter
    riesz_representer = TMLE.RieszRepresenter(Ψ)
    models = default_models()
    dataset = nothing ## dataset is ignored
    models[:riesz_representer] = riesz_learner()
    ## No sample split
    treatments_factor_estimator = TMLE.build_treatments_factor_estimator(
        riesz_representer, 
        models, 
        dataset; 
        train_validation_indices=nothing
    )
    @test treatments_factor_estimator isa TMLE.MLEstimator
    ## Sample split
    treatments_factor_estimator = TMLE.build_treatments_factor_estimator(
        riesz_representer, 
        models, 
        dataset; 
        train_validation_indices=[(1:50, 51:100), (51:100, 1:50)]
    )
    @test treatments_factor_estimator isa TMLE.SampleSplitMLEstimator
end

@testset "Test CMRelevantFactorsEstimator" begin
    dataset, _ = continuous_outcome_binary_treatment_pb()
    cache = Dict()
    models = default_models()
    models[:riesz_representer] = riesz_learner()
    η = TMLE.get_relevant_factors(Ψ, models)

    # With Riesz Learning: All fits are new
    models[:riesz_representer] = riesz_learner()
    riesz_η̂ = TMLE.CMRelevantFactorsEstimator(models=models)
    riesz_fit_log = [
        string("Required ", TMLE.string_repr(η)),
        TMLE.fit_string(η.treatments_factor),
        TMLE.fit_string(η.outcome_mean),
    ]
    test_logger = TestLogger()
    η̂ₙ = with_logger(test_logger) do
        riesz_η̂(η, dataset; cache=cache, verbosity=1)
    end
    info_logs = filter_warnings(test_logger)
    @test info_logs == riesz_fit_log
    @test η̂ₙ.outcome_mean isa TMLE.MLJEstimate{TMLE.ConditionalDistribution}
    @test η̂ₙ.treatments_factor isa TMLE.MLJEstimate{TMLE.RieszRepresenter{1, 3}}

    ## Changing the riesz model, Q unchanged
    models[:riesz_representer] = riesz_learner(hidden_layer=5)
    riesz_η̂ = TMLE.CMRelevantFactorsEstimator(models=models)
    Q_reused_fit_log = [
        string("Required ", TMLE.string_repr(η)),
        TMLE.fit_string(η.treatments_factor),
        TMLE.reuse_string(η.outcome_mean),
    ]
    test_logger = TestLogger()
    η̂ₙ = with_logger(test_logger) do
        riesz_η̂(η, dataset; cache=cache, verbosity=1)
    end
    info_logs = filter_warnings(test_logger)
    @test info_logs == Q_reused_fit_log

    ## Using sample splitting: both are refit
    train_validation_indices = MLJBase.train_test_pairs(CV(nfolds=3), 1:nrows(dataset), dataset)
    riesz_η̂_cv = TMLE.CMRelevantFactorsEstimator(models=models, train_validation_indices=train_validation_indices)
    test_logger = TestLogger()
    η̂ₙ = with_logger(test_logger) do
        riesz_η̂_cv(η, dataset; cache=cache, verbosity=1)
    end
    info_logs = filter_warnings(test_logger)
    @test info_logs == riesz_fit_log

    @test η̂ₙ.outcome_mean isa TMLE.SampleSplitMLJEstimate{TMLE.ConditionalDistribution}
    @test η̂ₙ.treatments_factor isa TMLE.SampleSplitMLJEstimate{TMLE.RieszRepresenter{1, 3}}
    ### Check predictions are made out of fold
    #### Outcome Mean
    ŷ = predict(η̂ₙ.outcome_mean, dataset)
    ŷ_expected = Vector{Normal{Float64}}(undef, 100)
    for (index, mach) in enumerate(η̂ₙ.outcome_mean.machines)
        _, val_idx = train_validation_indices[index]
        X = selectrows(TMLE.get_mlj_inputs(η̂ₙ.outcome_mean.estimand, dataset), val_idx)
        ŷ_expected[val_idx] = MLJBase.predict(mach, X)
    end
    @test getproperty.(ŷ, :μ) == getproperty.(ŷ_expected, :μ)
    @test getproperty.(ŷ, :σ) == getproperty.(ŷ_expected, :σ)
    #### Riesz Representer
    α̂ = predict(η̂ₙ.treatments_factor, dataset)
    α̂_expected = Vector{Float32}(undef, 100)
    for (index, mach) in enumerate(η̂ₙ.treatments_factor.machines)
        _, val_idx = train_validation_indices[index]
        X = selectrows(TMLE.get_mlj_inputs(η̂ₙ.treatments_factor.estimand, dataset), val_idx)
        α̂_expected[val_idx] = MLJBase.predict(mach, X)
    end
    @test α̂ ≈ α̂_expected
    #### clever covariate is the same as predict
    H, w = TMLE.clever_covariate_and_weights(Ψ, η̂ₙ.treatments_factor, dataset)
    @test H == α̂
    @test w == ones(100)
end

@testset "Test RieszRepresenterEstimator" begin
    # Define set of estimators for which the performance will be checked
    # using the RieszNetModel as the representer
    # we use a misspecified model to make sure inference is driven by the RieszRepresenter
    models = default_models(Q_continuous=MLJModels.DeterministicConstantRegressor())
    models[:riesz_representer] = riesz_learner(nepochs=500, batch_size=32)
    resampling = CausalStratifiedCV(StratifiedCV())
    riesz_estimators = (
        tmle = Tmle(models=models),
        ose = Ose(models=models),
        cv_tmle = Tmle(models=models, resampling=resampling),
        cv_ose = Ose(models=models, resampling=resampling),
    )

    Ψ₀ = 4
    ns = [100, 1000]
    nrepeats = 5
    rng = StableRNG(123)
    results = Dict(estimator_name => Dict(n => [] for n in ns)
        for estimator_name in keys(riesz_estimators)
    )
    
    cache = Dict()
    for n in ns
        for repeat in 1:nrepeats
            dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(;n=n, rng = rng)
            for (estimator_name, estimator) in pairs(riesz_estimators)
                @suppress begin
                    Ψ̂, cache = estimator(Ψ, dataset, verbosity=0);
                    push!(results[estimator_name][n], Ψ̂)
                end
            end
        end
    end
    
    for (estimator_name, estimator_results) in pairs(results)
        mean_bias_by_sample_size = [n => mean(abs.(TMLE.estimate.(n_results) .- Ψ₀)) for (n, n_results) in estimator_results]
        mean_bias_sorted_by_sample_size = last.(sort(mean_bias_by_sample_size, by=x->x[1]))
        for i in 1:length(ns)-1
            @test mean_bias_sorted_by_sample_size[i] >= mean_bias_sorted_by_sample_size[i+1]
        end
    end
end

@testset "Test C-TMLE and Riesz useful error message" begin
    models = default_models(Q_continuous=MLJModels.DeterministicConstantRegressor())
    models[:riesz_representer] = riesz_learner()
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(;)
    ctmle=Tmle(models=models, collaborative_strategy=AdaptiveCorrelationStrategy())
    @test_throws ArgumentError("C-TMLE does not support Riesz Learning yet.") ctmle(Ψ, dataset)
end

# TODO
# - [ ] Sort out the interface. estimator new field? Dispatch on G type for with encoder in models building?
# - [ ] Test Joint Estimat with Plugin

end

true
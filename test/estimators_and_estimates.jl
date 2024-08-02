module TestEstimatorsAndEstimates

using Test
using TMLE
using MLJBase
using DataFrames
using MLJGLMInterface
using MLJModels
using LogExpFunctions
using Distributions
using MLJLinearModels

verbosity = 1
n = 100
X, y = make_moons(n)
binary_dataset = DataFrame(Y=y, X₁=X.x1, X₂=X.x2)

X, y = make_regression(n)
continuous_dataset = DataFrame(Y=y, X₁=X.x1, X₂=X.x2)

estimand = TMLE.ConditionalDistribution(:Y, [:X₁, :X₂])
fit_log = string("Estimating: ", TMLE.string_repr(estimand))
reuse_log = string("Reusing estimate for: ", TMLE.string_repr(estimand))

@testset "Test MLConditionalDistributionEstimator: binary outcome" begin
    # Check predict / expected_value / compute_offset
    estimator = TMLE.MLConditionalDistributionEstimator(LinearBinaryClassifier())
    # Fitting with no cache
    cache = Dict()
    conditional_density_estimate = @test_logs (:info, fit_log) estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
    expected_features = collect(estimand.parents)
    @test conditional_density_estimate isa TMLE.MLConditionalDistribution
    @test fitted_params(conditional_density_estimate.machine).features == expected_features
    ŷ = predict(conditional_density_estimate, binary_dataset)
    mach_ŷ = predict(conditional_density_estimate.machine, binary_dataset[!, expected_features])
    @test all(ŷ[i].prob_given_ref == mach_ŷ[i].prob_given_ref for i in eachindex(ŷ))
    μ̂ = TMLE.expected_value(conditional_density_estimate, binary_dataset)
    @test μ̂ == [ŷ[i].prob_given_ref[2] for i in eachindex(ŷ)]
    offset = TMLE.compute_offset(conditional_density_estimate, binary_dataset)
    @test offset == logit.(μ̂)
    @test all(0. <= x <= 1. for x in TMLE.likelihood(conditional_density_estimate, binary_dataset)) # The pdf is not necessarily between 0 and 1
    # Check cache management
    ## Uses the cache instead of fitting
    new_estimator = TMLE.MLConditionalDistributionEstimator(LinearBinaryClassifier())
    @test TMLE.key(new_estimator) == TMLE.key(estimator)
    @test_logs (:info, reuse_log) estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
    ## Changing the model leads to refit
    new_estimator = TMLE.MLConditionalDistributionEstimator(LinearBinaryClassifier(fit_intercept=false))
    @test TMLE.key(new_estimator) != TMLE.key(estimator)
    @test_logs (:info, fit_log) new_estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
end

@testset "Test MLConditionalDistributionEstimator: continuous outcome" begin
    # Check predict / expected_value / compute_offset
    ## Probabilistic Model
    model = MLJGLMInterface.LinearRegressor()
    estimator = TMLE.MLConditionalDistributionEstimator(model)
    conditional_density_estimate = @test_logs (:info, fit_log) estimator(estimand, continuous_dataset; cache=Dict(), verbosity=verbosity)
    ŷ = predict(conditional_density_estimate, continuous_dataset)
    @test ŷ isa Vector{Normal{Float64}}
    μ̂ = TMLE.expected_value(conditional_density_estimate, continuous_dataset)
    @test [ŷᵢ.μ for ŷᵢ in ŷ] == μ̂
    offset = TMLE.compute_offset(conditional_density_estimate, continuous_dataset)
    @test offset == μ̂

    ## Deterministic Model
    model = MLJLinearModels.LinearRegressor()
    estimator = TMLE.MLConditionalDistributionEstimator(model)
    conditional_density_estimate = estimator(estimand, continuous_dataset; cache=Dict(), verbosity=0)
    ŷ = predict(conditional_density_estimate, continuous_dataset)
    @test ŷ isa Vector{Float64}
    μ̂ = TMLE.expected_value(conditional_density_estimate, continuous_dataset)
    @test ŷ == μ̂
    offset = TMLE.compute_offset(conditional_density_estimate, continuous_dataset)
    @test offset == μ̂
end

@testset "Test SampleSplitMLConditionalDistributionEstimator: Binary outcome" begin
    # Check predict / expected_value / compute_offset
    nfolds = 3
    train_validation_indices = Tuple(MLJBase.train_test_pairs(StratifiedCV(nfolds=nfolds), 1:n, binary_dataset, binary_dataset.Y))
    model = LinearBinaryClassifier()
    estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        model,
        train_validation_indices
    )
    cache = Dict()
    conditional_density_estimate = @test_logs (:info, fit_log) estimator(estimand, binary_dataset;cache=cache, verbosity=verbosity)
    @test conditional_density_estimate isa TMLE.SampleSplitMLConditionalDistribution
    expected_features = collect(estimand.parents)
    @test all(fitted_params(mach).features == expected_features for mach in conditional_density_estimate.machines)
    ŷ = predict(conditional_density_estimate, binary_dataset)
    @test ŷ isa UnivariateFiniteVector
    for foldid in 1:nfolds
        train, val = train_validation_indices[foldid]
        # The predictions on validation samples are made from
        # the machine trained on the train sample
        ŷfold = predict(conditional_density_estimate.machines[foldid], binary_dataset[val, expected_features])
        @test [ŷᵢ.prob_given_ref for ŷᵢ ∈ ŷ[val]] == [ŷᵢ.prob_given_ref for ŷᵢ ∈ ŷfold]
    end
    μ̂ = TMLE.expected_value(conditional_density_estimate, binary_dataset)
    @test [ŷᵢ.prob_given_ref[2] for ŷᵢ ∈ ŷ] == μ̂
    offset = TMLE.compute_offset(conditional_density_estimate, binary_dataset)
    @test offset == logit.(μ̂)
    @test all(0. <= x <= 1. for x in TMLE.likelihood(conditional_density_estimate, binary_dataset))
    # Check cache management
    ## Uses the cache instead of fitting
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        LinearBinaryClassifier(),
        train_validation_indices
    )
    @test TMLE.key(new_estimator) == TMLE.key(estimator)
    @test_logs (:info, reuse_log) estimator(estimand, binary_dataset;cache=cache, verbosity=verbosity)
    ## Changing the model leads to refit
    new_model = LinearBinaryClassifier(fit_intercept=false)
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        new_model,
        train_validation_indices
    )
    @test_logs (:info, fit_log) new_estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
    ## Changing the train/validation splits leads to refit
    train_validation_indices = Tuple(MLJBase.train_test_pairs(CV(nfolds=4), 1:n, binary_dataset))
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        new_model,
        train_validation_indices
    )
    @test_logs (:info, fit_log) new_estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
end

@testset "Test SampleSplitMLConditionalDistributionEstimator: Continuous outcome" begin
    # Check predict / expected_value / compute_offset
    nfolds = 3
    train_validation_indices = Tuple(MLJBase.train_test_pairs(CV(nfolds=nfolds), 1:n, continuous_dataset))
    ## Probabilistic Model
    model = MLJGLMInterface.LinearRegressor()
    estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        model,
        train_validation_indices
    )
    conditional_density_estimate = estimator(estimand, continuous_dataset; verbosity=0)
    ŷ = predict(conditional_density_estimate, continuous_dataset)
    @test ŷ isa Vector{Distributions.Normal{Float64}}
    μ̂ = TMLE.expected_value(conditional_density_estimate, continuous_dataset)
    @test μ̂ isa Vector{Float64}
    @test μ̂ == [ŷᵢ.μ for ŷᵢ in ŷ]
    offset = TMLE.compute_offset(conditional_density_estimate, continuous_dataset)
    @test offset == μ̂

    ## Deterministic Model
    model = MLJLinearModels.LinearRegressor()
    estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        model,
        train_validation_indices
    )
    conditional_density_estimate = estimator(estimand, continuous_dataset; verbosity=0)
    ŷ = predict(conditional_density_estimate, continuous_dataset)
    @test ŷ isa Vector{Float64}
    μ̂ = TMLE.expected_value(conditional_density_estimate, continuous_dataset)
    @test μ̂ == ŷ
    offset = TMLE.compute_offset(conditional_density_estimate, continuous_dataset)
    @test offset == μ̂
end
end

true
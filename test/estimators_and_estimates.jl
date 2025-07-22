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

@testset "Test MLEstimator: binary outcome" begin
    # Check predict / expected_value / compute_offset
    estimator = TMLE.MLEstimator(LinearBinaryClassifier())
    # Fitting with no cache
    cache = Dict()
    conditional_density_estimate = @test_logs (:info, fit_log) estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
    expected_features = collect(estimand.parents)
    @test conditional_density_estimate isa TMLE.MLJEstimate{TMLE.ConditionalDistribution}
    @test fitted_params(conditional_density_estimate.machine).features == expected_features
    ŷ = MLJBase.predict(conditional_density_estimate, binary_dataset)
    mach_ŷ = MLJBase.predict(conditional_density_estimate.machine, binary_dataset[!, expected_features])
    @test all(ŷ[i].prob_given_ref == mach_ŷ[i].prob_given_ref for i in eachindex(ŷ))
    μ̂ = TMLE.expected_value(conditional_density_estimate, binary_dataset)
    @test μ̂ == [ŷ[i].prob_given_ref[2] for i in eachindex(ŷ)]
    offset = TMLE.compute_offset(conditional_density_estimate, binary_dataset)
    @test offset == logit.(μ̂)
    @test all(0. <= x <= 1. for x in TMLE.likelihood(conditional_density_estimate, binary_dataset)) # The pdf is not necessarily between 0 and 1
    # Check cache management
    ## Uses the cache instead of fitting
    new_estimator = TMLE.MLEstimator(LinearBinaryClassifier())
    @test_logs (:info, reuse_log) estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
    ## Changing the model leads to refit
    new_estimator = TMLE.MLEstimator(LinearBinaryClassifier(fit_intercept=false))
    @test_logs (:info, fit_log) new_estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
end

@testset "Test MLEstimator: continuous outcome" begin
    # Check predict / expected_value / compute_offset
    ## Probabilistic Model
    model = MLJGLMInterface.LinearRegressor()
    estimator = TMLE.MLEstimator(model)
    conditional_density_estimate = @test_logs (:info, fit_log) estimator(estimand, continuous_dataset; cache=Dict(), verbosity=verbosity)
    ŷ = MLJBase.predict(conditional_density_estimate, continuous_dataset)
    @test ŷ isa Vector{Normal{Float64}}
    μ̂ = TMLE.expected_value(conditional_density_estimate, continuous_dataset)
    @test [ŷᵢ.μ for ŷᵢ in ŷ] == μ̂
    offset = TMLE.compute_offset(conditional_density_estimate, continuous_dataset)
    @test offset == μ̂

    ## Deterministic Model
    model = MLJLinearModels.LinearRegressor()
    estimator = TMLE.MLEstimator(model)
    conditional_density_estimate = estimator(estimand, continuous_dataset; cache=Dict(), verbosity=0)
    ŷ = MLJBase.predict(conditional_density_estimate, continuous_dataset)
    @test ŷ isa Vector{Float64}
    μ̂ = TMLE.expected_value(conditional_density_estimate, continuous_dataset)
    @test ŷ == μ̂
    offset = TMLE.compute_offset(conditional_density_estimate, continuous_dataset)
    @test offset == μ̂
end

@testset "Test SampleSplitMLEstimator: Binary outcome" begin
    # Check predict / expected_value / compute_offset
    nfolds = 3
    train_validation_indices = Tuple(MLJBase.train_test_pairs(StratifiedCV(nfolds=nfolds), 1:n, binary_dataset, binary_dataset.Y))
    model = LinearBinaryClassifier()
    estimator = TMLE.SampleSplitMLEstimator(
        model,
        train_validation_indices
    )
    cache = Dict()
    conditional_density_estimate = @test_logs (:info, fit_log) estimator(estimand, binary_dataset;cache=cache, verbosity=verbosity)
    @test conditional_density_estimate isa TMLE.SampleSplitMLJEstimate
    expected_features = collect(estimand.parents)
    @test all(fitted_params(mach).features == expected_features for mach in conditional_density_estimate.machines)
    ŷ = MLJBase.predict(conditional_density_estimate, binary_dataset)
    @test ŷ isa UnivariateFiniteVector
    for foldid in 1:nfolds
        train, val = train_validation_indices[foldid]
        # The predictions on validation samples are made from
        # the machine trained on the train sample
        ŷfold = MLJBase.predict(conditional_density_estimate.machines[foldid], binary_dataset[val, expected_features])
        @test [ŷᵢ.prob_given_ref for ŷᵢ ∈ ŷ[val]] == [ŷᵢ.prob_given_ref for ŷᵢ ∈ ŷfold]
    end
    μ̂ = TMLE.expected_value(conditional_density_estimate, binary_dataset)
    @test [ŷᵢ.prob_given_ref[2] for ŷᵢ ∈ ŷ] == μ̂
    offset = TMLE.compute_offset(conditional_density_estimate, binary_dataset)
    @test offset == logit.(μ̂)
    @test all(0. <= x <= 1. for x in TMLE.likelihood(conditional_density_estimate, binary_dataset))
    # Check cache management
    ## Uses the cache instead of fitting
    new_estimator = TMLE.SampleSplitMLEstimator(
        LinearBinaryClassifier(),
        train_validation_indices
    )
    @test_logs (:info, reuse_log) estimator(estimand, binary_dataset;cache=cache, verbosity=verbosity)
    ## Changing the model leads to refit
    new_model = LinearBinaryClassifier(fit_intercept=false)
    new_estimator = TMLE.SampleSplitMLEstimator(
        new_model,
        train_validation_indices
    )
    @test_logs (:info, fit_log) new_estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
    ## Changing the train/validation splits leads to refit
    train_validation_indices = Tuple(MLJBase.train_test_pairs(CV(nfolds=4), 1:n, binary_dataset))
    new_estimator = TMLE.SampleSplitMLEstimator(
        new_model,
        train_validation_indices
    )
    @test_logs (:info, fit_log) new_estimator(estimand, binary_dataset; cache=cache, verbosity=verbosity)
end

@testset "Test SampleSplitMLEstimator: Continuous outcome" begin
    # Check predict / expected_value / compute_offset
    nfolds = 3
    train_validation_indices = Tuple(MLJBase.train_test_pairs(CV(nfolds=nfolds), 1:n, continuous_dataset))
    ## Probabilistic Model
    model = MLJGLMInterface.LinearRegressor()
    estimator = TMLE.SampleSplitMLEstimator(
        model,
        train_validation_indices
    )
    conditional_density_estimate = estimator(estimand, continuous_dataset; verbosity=0)
    ŷ = MLJBase.predict(conditional_density_estimate, continuous_dataset)
    @test ŷ isa Vector{Distributions.Normal{Float64}}
    μ̂ = TMLE.expected_value(conditional_density_estimate, continuous_dataset)
    @test μ̂ isa Vector{Float64}
    @test μ̂ == [ŷᵢ.μ for ŷᵢ in ŷ]
    offset = TMLE.compute_offset(conditional_density_estimate, continuous_dataset)
    @test offset == μ̂

    ## Deterministic Model
    model = MLJLinearModels.LinearRegressor()
    estimator = TMLE.SampleSplitMLEstimator(
        model,
        train_validation_indices
    )
    conditional_density_estimate = estimator(estimand, continuous_dataset; verbosity=0)
    ŷ = MLJBase.predict(conditional_density_estimate, continuous_dataset)
    @test ŷ isa Vector{Float64}
    μ̂ = TMLE.expected_value(conditional_density_estimate, continuous_dataset)
    @test μ̂ == ŷ
    offset = TMLE.compute_offset(conditional_density_estimate, continuous_dataset)
    @test offset == μ̂
end

@testset "Test Conditional Distribution with no parents fits a marginal" begin
    binary_dataset = DataFrame(Y = categorical([1, 1, 0, 0, 0]))
    estimator = TMLE.MLEstimator(LinearBinaryClassifier())
    estimand = TMLE.ConditionalDistribution(:Y, ())
    estimate = estimator(estimand, binary_dataset, verbosity=0)
    @test estimate.machine.model isa ConstantClassifier
    ŷ = predict(estimate, binary_dataset)
    @test TMLE.expected_value(ŷ) == fill(0.4, 5)

    continuous_dataset = DataFrame(Y = [1., 2., 3., 4., 5.])
    @test_throws ArgumentError estimator(estimand, continuous_dataset,verbosity=0)

end

end

true
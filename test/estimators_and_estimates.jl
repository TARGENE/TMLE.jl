module TestEstimatorsAndEstimates

using Test
using TMLE
using MLJBase
using DataFrames
using MLJGLMInterface
using MLJModels
using LogExpFunctions
using Distributions

verbosity = 1
n = 100
X, y = make_moons(n)
dataset = DataFrame(Y=y, X₁=X.x1, X₂=X.x2)

X, y = make_regression(n)
continuous_dataset = DataFrame(Y=y, X₁=X.x1, X₂=X.x2)

estimand = TMLE.ConditionalDistribution(:Y, [:X₁, :X₂])
fit_log = string("Estimating: ", TMLE.string_repr(estimand))
reuse_log = string("Reusing estimate for: ", TMLE.string_repr(estimand))

@testset "Test MLConditionalDistributionEstimator" begin
    estimator = TMLE.MLConditionalDistributionEstimator(LinearBinaryClassifier())
    # Fitting with no cache
    cache = Dict()
    conditional_density_estimate = @test_logs (:info, fit_log) estimator(estimand, dataset; cache=cache, verbosity=verbosity)
    expected_features = collect(estimand.parents)
    @test conditional_density_estimate isa TMLE.MLConditionalDistribution
    @test fitted_params(conditional_density_estimate.machine).features == expected_features
    ŷ = predict(conditional_density_estimate, dataset)
    mach_ŷ = predict(conditional_density_estimate.machine, dataset[!, expected_features])
    @test all(ŷ[i].prob_given_ref == mach_ŷ[i].prob_given_ref for i in eachindex(ŷ))
    μ̂ = TMLE.expected_value(conditional_density_estimate, dataset)
    @test μ̂ == [ŷ[i].prob_given_ref[2] for i in eachindex(ŷ)]
    @test all(0. <= x <= 1. for x in TMLE.likelihood(conditional_density_estimate, dataset)) # The pdf is not necessarily between 0 and 1
    # Uses the cache instead of fitting
    new_estimator = TMLE.MLConditionalDistributionEstimator(LinearBinaryClassifier())
    @test TMLE.key(new_estimator) == TMLE.key(estimator)
    @test_logs (:info, reuse_log) estimator(estimand, dataset; cache=cache, verbosity=verbosity)
    # Changing the model leads to refit
    new_estimator = TMLE.MLConditionalDistributionEstimator(LinearBinaryClassifier(fit_intercept=false))
    @test TMLE.key(new_estimator) != TMLE.key(estimator)
    @test_logs (:info, fit_log) new_estimator(estimand, dataset; cache=cache, verbosity=verbosity)
end

@testset "Test SampleSplitMLConditionalDistributionEstimator" begin
    nfolds = 3
    train_validation_indices = Tuple(MLJBase.train_test_pairs(StratifiedCV(nfolds=nfolds), 1:n, dataset, dataset.Y))
    model = LinearBinaryClassifier()
    estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        model,
        train_validation_indices
    )
    cache = Dict()
    conditional_density_estimate = @test_logs (:info, fit_log) estimator(estimand, dataset;cache=cache, verbosity=verbosity)
    @test conditional_density_estimate isa TMLE.SampleSplitMLConditionalDistribution
    expected_features = collect(estimand.parents)
    @test all(fitted_params(mach).features == expected_features for mach in conditional_density_estimate.machines)
    ŷ = predict(conditional_density_estimate, dataset)
    @test ŷ isa UnivariateFiniteVector
    μ̂ = TMLE.expected_value(conditional_density_estimate, dataset)
    for foldid in 1:nfolds
        train, val = train_validation_indices[foldid]
        # The predictions on validation samples are made from
        # the machine trained on the train sample
        ŷfold = predict(conditional_density_estimate.machines[foldid], dataset[val, expected_features])
        @test [ŷᵢ.prob_given_ref for ŷᵢ ∈ ŷ[val]] == [ŷᵢ.prob_given_ref for ŷᵢ ∈ ŷfold]
        @test μ̂[val] == [ŷᵢ.prob_given_ref[2] for ŷᵢ ∈ ŷfold]
    end
    @test all(0. <= x <= 1. for x in TMLE.likelihood(conditional_density_estimate, dataset))
    # Uses the cache instead of fitting
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        LinearBinaryClassifier(),
        train_validation_indices
    )
    @test TMLE.key(new_estimator) == TMLE.key(estimator)
    @test_logs (:info, reuse_log) estimator(estimand, dataset;cache=cache, verbosity=verbosity)
    # Changing the model leads to refit
    new_model = LinearBinaryClassifier(fit_intercept=false)
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        new_model,
        train_validation_indices
    )
    @test_logs (:info, fit_log) new_estimator(estimand, dataset; cache=cache, verbosity=verbosity)
    # Changing the train/validation splits leads to refit
    train_validation_indices = Tuple(MLJBase.train_test_pairs(CV(nfolds=4), 1:n, dataset))
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        new_model,
        train_validation_indices
    )
    @test_logs (:info, fit_log) new_estimator(estimand, dataset; cache=cache, verbosity=verbosity)
end

@testset "Test SampleSplitMLConditionalDistributionEstimator: Continuous outcome" begin
    nfolds = 3
    train_validation_indices = Tuple(MLJBase.train_test_pairs(CV(nfolds=nfolds), 1:n, continuous_dataset))
    model = MLJGLMInterface.LinearRegressor()
    estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        model,
        train_validation_indices
    )
    conditional_density_estimate = estimator(estimand, continuous_dataset; verbosity=verbosity)
    ŷ = predict(conditional_density_estimate, continuous_dataset)
    @test ŷ isa Vector{Distributions.Normal{Float64}}
    μ̂ = TMLE.expected_value(conditional_density_estimate, continuous_dataset)
    @test μ̂ isa Vector{Float64}
end

@testset "Test compute_offset MLConditionalDistributionEstimator" begin
    dataset = (
        T = categorical(["a", "b", "c", "a", "a", "b", "a"]),
        Ycont = [1., 2., 3, 4, 5, 6, 7],
        Ycat = categorical([1, 0, 0, 1, 1, 1, 0]),
        W = rand(7),
    )
    μYcont = mean(dataset.Ycont)
    μYcat = mean(float(dataset.Ycat))
    # The model is probabilistic continuous, the offset is the mean of 
    # the conditional distribution
    conditional_density_estimate = TMLE.MLConditionalDistributionEstimator(ConstantRegressor())(
        TMLE.ConditionalDistribution(:Ycont, [:W, :T]),
        dataset,
        verbosity=0
    )
    offset = TMLE.compute_offset(conditional_density_estimate, dataset)
    @test offset == mean.(predict(conditional_density_estimate, dataset))
    @test offset == repeat([μYcont], 7)
    # The model is deterministic, the offset is simply the output 
    # of the predict function which is assumed to correspond to the mean
    # if the squared loss was optimized for by the underlying model
    conditional_density_estimate = TMLE.MLConditionalDistributionEstimator(DeterministicConstantRegressor())(
        TMLE.ConditionalDistribution(:Ycont, [:W, :T]),
        dataset,
        verbosity=0
    )
    offset = TMLE.compute_offset(conditional_density_estimate, dataset)
    @test offset == predict(conditional_density_estimate, dataset)
    @test offset == repeat([μYcont], 7)
    # The model is probabilistic binary, the offset is the logit
    # of the mean of the conditional distribution
    conditional_density_estimate = TMLE.MLConditionalDistributionEstimator(ConstantClassifier())(
        TMLE.ConditionalDistribution(:Ycat, [:W, :T]),
        dataset,
        verbosity=0
    )
    offset = TMLE.compute_offset(conditional_density_estimate, dataset)
    @test offset == repeat([logit(μYcat)], 7)
end

end

true
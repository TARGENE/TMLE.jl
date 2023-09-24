module TestEstimatorsAndEstimates

using Test
using TMLE
using MLJBase
using DataFrames
using MLJGLMInterface
using MLJModels
using LogExpFunctions

verbosity = 1
n = 100
X, y = make_moons(n)
dataset = DataFrame(Y=y, X₁=X.x1, X₂=X.x2)
estimand = ConditionalDistribution(:Y, [:X₁, :X₂])
fit_log = string("Estimating: ", TMLE.string_repr(estimand))
reuse_log = string("Reusing estimate for: ", TMLE.string_repr(estimand))

@testset "Test MLConditionalDistributionEstimator" begin
    estimator = TMLE.MLConditionalDistributionEstimator(LinearBinaryClassifier())
    # Fitting with no cache
    cache = Dict()
    estimate = @test_logs (:info, fit_log) estimator(estimand, dataset; cache=cache, verbosity=verbosity)
    expected_features = collect(estimand.parents)
    @test estimate isa TMLE.MLConditionalDistribution
    @test fitted_params(estimate.machine).features == expected_features
    ŷ = predict(estimate, dataset)
    mach_ŷ = predict(estimate.machine, dataset[!, expected_features])
    @test all(ŷ[i].prob_given_ref == mach_ŷ[i].prob_given_ref for i in eachindex(ŷ))
    μ̂ = TMLE.expected_value(estimate, dataset)
    @test μ̂ == [ŷ[i].prob_given_ref[2] for i in eachindex(ŷ)]
    @test all(0. <= x <= 1. for x in TMLE.likelihood(estimate, dataset)) # The pdf is not necessarily between 0 and 1
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
    train_validation_indices = collect(MLJBase.train_test_pairs(CV(nfolds=nfolds), 1:n, dataset))
    model = LinearBinaryClassifier()
    estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        model,
        train_validation_indices
    )
    cache = Dict()
    estimate = @test_logs (:info, fit_log) estimator(estimand, dataset;cache=cache, verbosity=verbosity)
    @test estimate isa TMLE.SampleSplitMLConditionalDistribution
    expected_features = collect(estimand.parents)
    @test all(fitted_params(mach).features == expected_features for mach in estimate.machines)
    ŷ = predict(estimate, dataset)
    μ̂ = TMLE.expected_value(estimate, dataset)
    for foldid in 1:nfolds
        train, val = train_validation_indices[foldid]
        # The predictions on validation samples are made from
        # the machine trained on the train sample
        ŷfold = predict(estimate.machines[foldid], dataset[val, expected_features])
        @test [ŷᵢ.prob_given_ref for ŷᵢ ∈ ŷ[val]] == [ŷᵢ.prob_given_ref for ŷᵢ ∈ ŷfold]
        @test μ̂[val] == [ŷᵢ.prob_given_ref[2] for ŷᵢ ∈ ŷfold]
    end
    @test all(0. <= x <= 1. for x in TMLE.likelihood(estimate, dataset))
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
    train_validation_indices = collect(MLJBase.train_test_pairs(CV(nfolds=4), 1:n, dataset))
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        new_model,
        train_validation_indices
    )
    @test_logs (:info, fit_log) new_estimator(estimand, dataset; cache=cache, verbosity=verbosity)
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
    distr_estimate = TMLE.MLConditionalDistributionEstimator(ConstantRegressor())(
        TMLE.ConditionalDistribution(:Ycont, [:W, :T]),
        dataset,
        verbosity=0
    )
    offset = TMLE.compute_offset(distr_estimate, dataset)
    @test offset == mean.(predict(distr_estimate, dataset))
    @test offset == repeat([μYcont], 7)
    # The model is deterministic, the offset is simply the output 
    # of the predict function which is assumed to correspond to the mean
    # if the squared loss was optimized for by the underlying model
    distr_estimate = TMLE.MLConditionalDistributionEstimator(DeterministicConstantRegressor())(
        TMLE.ConditionalDistribution(:Ycont, [:W, :T]),
        dataset,
        verbosity=0
    )
    offset = TMLE.compute_offset(distr_estimate, dataset)
    @test offset == predict(distr_estimate, dataset)
    @test offset == repeat([μYcont], 7)
    # The model is probabilistic binary, the offset is the logit
    # of the mean of the conditional distribution
    distr_estimate = TMLE.MLConditionalDistributionEstimator(ConstantClassifier())(
        TMLE.ConditionalDistribution(:Ycat, [:W, :T]),
        dataset,
        verbosity=0
    )
    offset = TMLE.compute_offset(distr_estimate, dataset)
    @test offset == repeat([logit(μYcat)], 7)
end

end

true
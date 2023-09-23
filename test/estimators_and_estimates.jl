module TestEstimatorsAndEstimates

using Test
using TMLE
using MLJBase
using DataFrames
using MLJGLMInterface

verbosity = 1
n = 100
X, y = make_regression(n, 4)
dataset = DataFrame(Y=y, T₁=X.x1, T₂=X.x2, W=X.x3)
estimand = ConditionalDistribution(:Y, [:W, :T₁, :T₂])
fit_log = string("Estimating: ", TMLE.string_repr(estimand))
reuse_log = string("Reusing estimate for: ", TMLE.string_repr(estimand))

@testset "Test MLConditionalDistributionEstimator" begin
    estimator = TMLE.MLConditionalDistributionEstimator(LinearRegressor())
    # Fitting with no cache
    cache = Dict()
    estimate = @test_logs (:info, fit_log) estimator(estimand, dataset; cache=cache, verbosity=verbosity)
    expected_features = collect(estimand.parents)
    @test estimate isa TMLE.MLConditionalDistribution
    @test fitted_params(estimate.machine).features == expected_features
    ŷ = predict(estimate, dataset)
    @test ŷ == predict(estimate.machine, dataset[!, expected_features])
    μ̂ = TMLE.expected_value(estimate, dataset)
    @test μ̂ == mean.(ŷ)
    @test all(0. <= x <= 1. for x in TMLE.likelihood(estimate, dataset))
    # Uses the cache instead of fitting
    new_estimator = TMLE.MLConditionalDistributionEstimator(LinearRegressor())
    @test TMLE.key(new_estimator) == TMLE.key(estimator)
    @test_logs (:info, reuse_log) estimator(estimand, dataset; cache=cache, verbosity=verbosity)
    # Changing the model leads to refit
    new_estimator = TMLE.MLConditionalDistributionEstimator(LinearRegressor(fit_intercept=false))
    @test_logs (:info, fit_log) new_estimator(estimand, dataset; cache=cache, verbosity=verbosity)
    # The cache contains two estimators for the estimand
    @test length(cache) == 2
end

@testset "Test SampleSplitMLConditionalDistributionEstimator" begin
    nfolds = 3
    train_validation_indices = collect(MLJBase.train_test_pairs(CV(nfolds=nfolds), 1:n, dataset))
    model = LinearRegressor()
    estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        LinearRegressor(),
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
        @test ŷ[val] == ŷfold
        @test μ̂[val] == mean.(ŷfold)
    end
    all(0. <= x <= 1. for x in TMLE.likelihood(estimate, dataset))
    # Uses the cache instead of fitting
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        LinearRegressor(),
        train_validation_indices
    )
    @test TMLE.key(new_estimator) == TMLE.key(estimator)
    @test_logs (:info, reuse_log) estimator(estimand, dataset;cache=cache, verbosity=verbosity)
    # Changing the model leads to refit
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        LinearRegressor(fit_intercept=false),
        train_validation_indices
    )
    @test_logs (:info, fit_log) new_estimator(estimand, dataset; cache=cache, verbosity=verbosity)
    # Changing the train/validation splits leads to refit
    train_validation_indices = collect(MLJBase.train_test_pairs(CV(nfolds=4), 1:n, dataset))
    new_estimator = TMLE.SampleSplitMLConditionalDistributionEstimator(
        LinearRegressor(),
        train_validation_indices
    )
    @test_logs (:info, fit_log) new_estimator(estimand, dataset; cache=cache, verbosity=verbosity)
    # The cache contains 3 estimators for the estimand
    @test length(cache) == 3
end

end

true
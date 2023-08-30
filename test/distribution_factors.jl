module TestDistributionFactors

using Test
using TMLE
using MLJGLMInterface
using MLJBase

@testset "Test ConditionalDistribution(outcome, parents, model, resampling)" begin
    # Default constructor
    cond_dist = ConditionalDistribution("Y", [:W, "T", :C], LinearBinaryClassifier())
    @test cond_dist.outcome == :Y
    @test cond_dist.parents == Set([:W, :T, :C])
    @test cond_dist.model == LinearBinaryClassifier()
    @test TMLE.key(cond_dist) == (:Y, Set([:W, :T, :C]))
    # String representation
    @test TMLE.string_repr(cond_dist) == "Y | T, W, C"
end

@testset "Test fit!" begin
    X, y = make_regression()
    dataset = (Y = y, X₁ = X.x1, X₂ = X.x2)
    cond_dist = ConditionalDistribution(:Y, [:X₁, :X₂], LinearRegressor())
    # Initial fit
    log_sequence = (
        (:info, TMLE.fit_message(cond_dist)),
        (:info, "Training machine(LinearRegressor(fit_intercept = true, …), …).")
    )
    @test_logs log_sequence... MLJBase.fit!(cond_dist, dataset, verbosity=2)
    fp = fitted_params(cond_dist.machine)
    @test fp.features == [:X₂, :X₁]
    @test fp.intercept != 0.
    # Change the model
    cond_dist.model = LinearRegressor(fit_intercept=false)
    log_sequence = (
        (:info, TMLE.fit_message(cond_dist)),
        (:info, "Training machine(LinearRegressor(fit_intercept = false, …), …).")
    )
    @test_logs log_sequence... MLJBase.fit!(cond_dist, dataset, verbosity=2)
    fp = fitted_params(cond_dist.machine)
    @test fp.features == [:X₂, :X₁]
    @test fp.intercept == 0.
    # Change model's hyperparameter
    cond_dist.model.fit_intercept = true
    log_sequence = (
        (:info, TMLE.update_message(cond_dist)),
        (:info, "Updating machine(LinearRegressor(fit_intercept = true, …), …).")
    )
    @test_logs log_sequence... MLJBase.fit!(cond_dist, dataset, verbosity=2)
    fp = fitted_params(cond_dist.machine)
    @test fp.features == [:X₂, :X₁]
    @test fp.intercept != 0.
    # Forcing refit
    log_sequence = (
        (:info, TMLE.update_message(cond_dist)),
        (:info, "Training machine(LinearRegressor(fit_intercept = true, …), …).")
    )
    @test_logs log_sequence... MLJBase.fit!(cond_dist, dataset, verbosity=2, force = true)
    # No refit
    log_sequence = (
        (:info, TMLE.update_message(cond_dist)),
        (:info, "Not retraining machine(LinearRegressor(fit_intercept = true, …), …). Use `force=true` to force.")
    )
    @test_logs log_sequence... MLJBase.fit!(cond_dist, dataset, verbosity=2)

end

end

true
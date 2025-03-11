using TMLE
using Test
using MLJBase
using CategoricalArrays
using StatisticalMeasures

"""
The risk for continuous outcomes is the MSE
"""
risk(ŷ, y) = rmse(TMLE.expected_value(ŷ), y)

"""
The risk for binary outcomes is the LogLoss
"""
risk(ŷ, y::CategoricalArray) = mean(log_loss(ŷ, y))

"""
    test_fluct_decreases_risk(cache; outcome_name::Symbol=nothing)

The fluctuation is supposed to decrease the risk as its objective function is the risk itself.
It seems that sometimes this is not entirely true in practice, so the test actually checks that it does not
increase risk more than tol
"""
function test_fluct_decreases_risk(cache; atol=1e-6)
    fluctuated_mean_machine = cache[:targeted_factors].outcome_mean.machine
    initial_mean_machine = fluctuated_mean_machine.model.initial_factors.outcome_mean.machine
    y = fluctuated_mean_machine.data[2]
    initial_risk = risk(MLJBase.predict(initial_mean_machine), y)
    fluct_risk = risk(MLJBase.predict(fluctuated_mean_machine), y)
    @test initial_risk >= fluct_risk || isapprox(initial_risk, fluct_risk, atol=atol)
end


"""
    test_coverage(tmle_result::TMLE.EICEstimate, Ψ₀)

Both the TMLE and OneStep estimators are suppose to be asymptotically efficient and cover the truth 
at the given confidence level: here 0.05
"""
function test_coverage(result::TMLE.EICEstimate, Ψ₀)
    # TMLE
    lb, ub = confint(significance_test(result))
    @test lb ≤ Ψ₀ ≤ ub
end

"""
    test_mean_inf_curve_almost_zero(tmle_result::TMLE.EICEstimate; atol=1e-10)

The TMLE is supposed to solve the EIC score equation.
"""
test_mean_inf_curve_almost_zero(tmle_result::TMLE.EICEstimate; atol=1e-10) = @test mean(tmle_result.IC) ≈ 0.0 atol=atol

double_robust_estimators(models; resampling=CV(nfolds=3)) = (
    tmle = TMLEE(models=models, machine_cache=true),
    ose = OSE(models=models, machine_cache=true),
    cv_tmle = TMLEE(models=models, resampling=resampling, machine_cache=true),
    cv_ose = TMLEE(models=models, resampling=resampling, machine_cache=true),
)

function test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    cache = Dict()
    results = []
    for estimator ∈ dr_estimators
        result, cache = estimator(Ψ, dataset, cache=cache, verbosity=verbosity)
        push!(results, result)
        test_coverage(result, Ψ₀)
        if estimator isa TMLEE && estimator.resampling === nothing
            test_fluct_decreases_risk(cache)
        end
    end
    return NamedTuple{keys(dr_estimators)}(results), cache
end
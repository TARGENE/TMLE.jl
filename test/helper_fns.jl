using TMLE
using Test
using MLJBase
using CategoricalArrays

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
    fluctuated_mean_machine = cache[:last_fluctuation].outcome_mean.machine
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
    lb, ub = confint(OneSampleTTest(result))
    @test lb ≤ Ψ₀ ≤ ub
    # OneStep
    lb, ub = confint(OneSampleZTest(result))
    @test lb ≤ Ψ₀ ≤ ub
end

"""
    test_mean_inf_curve_almost_zero(tmle_result::TMLE.EICEstimate; atol=1e-10)

The TMLE is supposed to solve the EIC score equation.
"""
test_mean_inf_curve_almost_zero(tmle_result::TMLE.EICEstimate; atol=1e-10) = @test mean(tmle_result.IC) ≈ 0.0 atol=atol

"""
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result::TMLE.TMLEResult)

This cqnnot be guaranteed in general since a well specified maximum 
likelihood estimator also solves the score equation.
"""
test_fluct_mean_inf_curve_lower_than_initial(tmle_result::TMLE.TMLEstimate, ose_result::TMLE.OSEstimate) = @test abs(mean(tmle_result.IC)) < abs(mean(ose_result.IC))

double_robust_estimators(models; resampling=CV(nfolds=3)) = (
    tmle = TMLEE(models),
    ose = OSE(models),
    cv_tmle = TMLEE(models, resampling=resampling),
    cv_ose = TMLEE(models, resampling=resampling),
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
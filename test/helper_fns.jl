using TMLE
using Test
using MLJBase
using CategoricalArrays

"""
The risk for continuous targets is the MSE
"""
risk(ŷ, y) = rmse(TMLE.expected_value(ŷ), y)

"""
The risk for binary targets is the LogLoss
"""
risk(ŷ, y::CategoricalArray) = mean(log_loss(ŷ, y))

"""
    test_fluct_decreases_risk(cache; target_name::Symbol=nothing)

The fluctuation is supposed to decrease the risk as its objective function is the risk itself.
It seems that sometimes this is not entirely true in practice, so the test actually checks that it does not
increase risk more than tol
"""
function test_fluct_decreases_risk(cache; target_name::Symbol=nothing, atol=1e-6)
    y = cache.data[:no_missing][target_name]
    initial_risk = risk(cache.data[:Q₀], y)
    fluct_risk = risk(cache.data[:Qfluct], y)
    @test initial_risk >= fluct_risk || isapprox(initial_risk, fluct_risk, atol=atol)
end


"""
    test_coverage(tmle_result::TMLE.TMLEResult, Ψ₀)

Both the TMLE and OneStep estimators are suppose to be asymptotically efficient and cover the truth 
at the given confidence level: here 0.05
"""
function test_coverage(tmle_result::TMLE.TMLEResult, Ψ₀)
    # TMLE
    lb, ub = confint(OneSampleTTest(tmle_result.tmle))
    @test lb ≤ Ψ₀ ≤ ub
    # OneStep
    lb, ub = confint(OneSampleTTest(tmle_result.onestep))
    @test lb ≤ Ψ₀ ≤ ub
end

"""
    test_mean_inf_curve_almost_zero(tmle_result::TMLE.TMLEResult; atol=1e-10)

The TMLE is supposed to solve the EIC score equation.
"""
test_mean_inf_curve_almost_zero(tmle_result::TMLE.TMLEResult; atol=1e-10) = @test mean(tmle_result.tmle.IC) ≈ 0.0 atol=atol

"""
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result::TMLE.TMLEResult)

This cqnnot be guaranteed in general since a well specified maximum 
likelihood estimator also solves the score equation.
"""
test_fluct_mean_inf_curve_lower_than_initial(tmle_result::TMLE.TMLEResult) = @test abs(mean(tmle_result.tmle.IC)) < abs(mean(tmle_result.onestep.IC))
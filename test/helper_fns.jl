using TMLE
using Test
using MLJBase
using CategoricalArrays

risk(ŷ, y) = rmse(TMLE.expected_value(ŷ), y)
risk(ŷ, y::CategoricalArray) = mean(log_loss(ŷ, y))

function test_fluct_decreases_risk(cache; target_name::Symbol=nothing)
    y = cache.data[:no_missing][target_name]
    initial_risk = risk(cache.data[:Q₀], y)
    fluct_risk = risk(cache.data[:Qfluct], y)
    @test initial_risk > fluct_risk
end

function test_fluct_risk_almost_equal_to_initial(cache; target_name::Symbol=nothing, atol=1e-6)
    y = cache.data[:no_missing][target_name]
    initial_risk = risk(cache.data[:Q₀], y)
    fluct_risk = risk(cache.data[:Qfluct], y)
    @test initial_risk ≈ fluct_risk atol=atol
end

function test_coverage(tmle_result::TMLE.TMLEResult, Ψ₀)
    # TMLE
    lb, ub = confint(OneSampleTTest(tmle_result.tmle))
    @test lb ≤ Ψ₀ ≤ ub
    # OneStep
    lb, ub = confint(OneSampleTTest(tmle_result.onestep))
    @test lb ≤ Ψ₀ ≤ ub
end


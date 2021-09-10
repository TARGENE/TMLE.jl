abstract type TMLEstimator <: MLJ.Model end

"""

Let's default to no warnings for now.
"""
MLJBase.check(model::TMLEstimator, args... ; full=false) = true

pvalue(tmle::TMLEstimator, estimate, stderror) = 2*(1 - cdf(Normal(0, 1), abs(estimate/stderror)))

confint(tmle::TMLEstimator, estimate, stderror) = (estimate - 1.96stderror, estimate + 1.96stderror)
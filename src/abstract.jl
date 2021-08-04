abstract type TMLEstimator end

pvalue(tmle::TMLEstimator, estimate, stderror) = 2*(1 - cdf(Normal(0, 1), abs(estimate/stderror)))

confint(tmle::TMLEstimator, estimate, stderror) = (estimate - 1.96stderror, estimate + 1.96stderror)
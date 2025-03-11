
MLJBase.report(factors::MLCMRelevantFactors) = MLJBase.report(factors.outcome_mean.machine)

MLJBase.report(cache) = MLJBase.report(cache[:targeted_factors])

"""
    gradients(cache)

Retrieves the gradients corresponding to each targeting step from the cache.
"""
gradients(cache) = MLJBase.report(cache[:targeted_factors]).gradients

"""
    estimates(cache)

Retrieves the estimates corresponding to each targeting step from the cache.
"""
estimates(cache) = MLJBase.report(cache[:targeted_factors]).estimates

"""
    epsilons(cache)

Retrieves the fluctuations' epsilons corresponding to each targeting step from the cache.
"""
epsilons(cache) = MLJBase.report(cache[:targeted_factors]).epsilons

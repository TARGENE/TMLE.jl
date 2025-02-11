
MLJBase.report(factors::MLCMRelevantFactors) = MLJBase.report(factors.outcome_mean.machine)

MLJBase.report(cache) = MLJBase.report(cache[:targeted_factors])

gradients(cache) = MLJBase.report(cache[:targeted_factors]).gradients

estimates(cache) = MLJBase.report(cache[:targeted_factors]).estimates

epsilons(cache) = MLJBase.report(cache[:targeted_factors]).epsilons

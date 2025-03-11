function targeting(model, dataset; verbosity=1, machine_cache=false)
    outcome_mean = model.initial_factors.outcome_mean.estimand

    fluctuated_estimator = MLConditionalDistributionEstimator(model)
    fluctuated_outcome_mean = try
        fluctuated_estimator(
            outcome_mean,
            dataset,
            verbosity=verbosity,
            machine_cache=machine_cache
        )
    catch e
        throw(FitFailedError(outcome_mean, model, outcome_mean_fluctuation_fit_error_msg(outcome_mean), e))
    end
    # Do not fluctuate propensity score
    return model.initial_factors.propensity_score, fluctuated_outcome_mean
end

function targeting(model::Vector, dataset; verbosity=1, machine_cache=false)
    for model in models

    end
end

function targeting(model::MLJTuning.DeterministicTunedModel, dataset; verbosity=1, machine_cache=false)

end
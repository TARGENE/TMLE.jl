abstract type CollaborativeStrategy end

#####################################################################
###                  CMRelevantFactorsEstimator                   ###
#####################################################################

struct FitFailedError <: Exception
    estimand::Estimand
    msg::String
    origin::Exception
end

default_fit_error_msg(factor) = string(
    "Could not fit the following model: ", 
    string_repr(factor), 
    ".\n Hint: don't forget to use `with_encoder` to encode categorical variables.")

propensity_score_fit_error_msg(factor) = string("Could not fit the following propensity score model: ", string_repr(factor))

outcome_mean_fit_error_msg(factor) = string(
    "Could not fit the following Outcome mean model: ", 
    string_repr(factor), 
    ".\n Hint: don't forget to use `with_encoder` to encode categorical variables.")

outcome_mean_fluctuation_fit_error_msg(factor) = string(
    "Could not fluctuate the following Outcome mean: ", 
    string_repr(factor), 
    ".")

Base.showerror(io::IO, e::FitFailedError) = print(io, e.msg)

struct CMRelevantFactorsEstimator <: Estimator
    resampling::Union{Nothing, ResamplingStrategy}
    models::Dict
end

CMRelevantFactorsEstimator(;models, resampling=nothing) = CMRelevantFactorsEstimator(resampling, models)

key(estimator::CMRelevantFactorsEstimator) = 
    (CMRelevantFactorsEstimator, estimator.resampling, estimator.models)

get_train_validation_indices(resampling, factors, dataset) = nothing

function get_train_validation_indices(resampling::ResamplingStrategy, factors, dataset)
    relevant_columns = collect(variables(factors))
    outcome_variable = factors.outcome_mean.outcome
    feature_variables = filter(x -> x !== outcome_variable, relevant_columns)
    return Tuple(MLJBase.train_test_pairs(
        resampling,
        1:nrows(dataset),
        selectcols(dataset, feature_variables), 
        Tables.getcolumn(dataset, outcome_variable)
    ))
end

function acquire_model(models, key, dataset, is_propensity_score)
    # If the model is in models return it
    haskey(models, key) && return models[key]
    # Otherwise, if the required model is for a propensity_score, return the default
    model_default = :G_default
    if !is_propensity_score
        # Finally, if the required model is an outcome_mean, find the type from the data
        model_default = is_binary(dataset, key) ? :Q_binary_default : :Q_continuous_default
    end
    return models[model_default]
end

function build_propensity_score_estimator(propensity_score, models, dataset;
    train_validation_indices=nothing,
    )
    cd_estimators = Dict()
    for conditional_distribution in propensity_score
        outcome = conditional_distribution.outcome
        model = acquire_model(models, outcome, dataset, true)
        cd_estimators[outcome] = ConditionalDistributionEstimator(model, train_validation_indices)
    end
    return JointConditionalDistributionEstimator(cd_estimators)
end

function estimate_propensity_score(propensity_score, models, dataset;
    train_validation_indices=nothing,
    cache=Dict(),
    verbosity=1,
    machine_cache=false
    )
    propensity_score_estimator = build_propensity_score_estimator(
        propensity_score, 
        models,  
        dataset;
        train_validation_indices=train_validation_indices,
    )
    return propensity_score_estimator(
        propensity_score, 
        dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
end

function estimate_outcome_mean(outcome_mean, models, dataset;
    train_validation_indices=nothing,
    cache=Dict(),
    verbosity=1,
    machine_cache=false
    )
    outcome_model = acquire_model(models, outcome_mean.outcome, dataset, false)
    outcome_mean_estimator = ConditionalDistributionEstimator(
        outcome_model,
        train_validation_indices, 
    )
    return try_fit_ml_estimator(outcome_mean_estimator, outcome_mean, dataset;
        error_fn=outcome_mean_fit_error_msg,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
end

function (estimator::CMRelevantFactorsEstimator)(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    if haskey(cache, estimand)
        old_estimator, estimate = cache[estimand]
        if key(old_estimator) == key(estimator)
            verbosity > 0 && @info(reuse_string(estimand))
            return estimate
        end
    end
    verbosity > 0 && @info(string("Required ", string_repr(estimand)))
    models = estimator.models
    outcome_mean = estimand.outcome_mean
    propensity_score = estimand.propensity_score
    resampling = estimator.resampling
    # Get train validation indices
    train_validation_indices = get_train_validation_indices(resampling, estimand, dataset)
    # Estimate propensity score
    propensity_score_estimate = estimate_propensity_score(propensity_score, models, dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # Estimate outcome mean
    outcome_mean_estimate = estimate_outcome_mean(outcome_mean, models, dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, outcome_mean_estimate, propensity_score_estimate)
    # Update cache
    cache[estimand] = estimator => estimate

    return estimate
end

#####################################################################
###              TargetedCMRelevantFactorsEstimator               ###
#####################################################################

struct TargetedCMRelevantFactorsEstimator{S <: Union{Nothing, CollaborativeStrategy}}
    fluctuation::Fluctuation
    collaborative_strategy::S
end

function TargetedCMRelevantFactorsEstimator(Ψ, initial_factors_estimate; 
    collaborative_strategy::S=nothing, 
    tol=nothing, 
    max_iter=1, 
    ps_lowerbound=1e-8, 
    weighted=false, 
    machine_cache=false
    ) where S <: Union{Nothing, CollaborativeStrategy}
    fluctuation_model = Fluctuation(Ψ, initial_factors_estimate; 
        tol=tol,
        max_iter=max_iter, 
        ps_lowerbound=ps_lowerbound, 
        weighted=weighted,
        cache=machine_cache
    )
    return TargetedCMRelevantFactorsEstimator{S}(fluctuation_model, collaborative_strategy)
end

"""

Targeted estimator in the absence of a collaborative strategy.
"""
function (estimator::TargetedCMRelevantFactorsEstimator{Nothing})(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    fluctuation_model = estimator.fluctuation
    outcome_mean = fluctuation_model.initial_factors.outcome_mean.estimand
    # Fluctuate outcome model
    fluctuated_estimator = MLConditionalDistributionEstimator(fluctuation_model)
    fluctuated_outcome_mean = try_fit_ml_estimator(fluctuated_estimator, outcome_mean, dataset;
        error_fn=outcome_mean_fluctuation_fit_error_msg,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # Do not fluctuate propensity score
    fluctuated_propensity_score = fluctuation_model.initial_factors.propensity_score
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, fluctuated_outcome_mean, fluctuated_propensity_score)
    # Update cache
    cache[:targeted_factors] = estimate

    return estimate
end

#####################################################################
###                  CMRelevantFactorsEstimator                   ###
#####################################################################

struct FitFailedError <: Exception
    estimand::Estimand
    model::MLJBase.Model
    msg::String
    origin::Exception
end

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

struct CMRelevantFactorsEstimator{S <: Union{Nothing, CollaborativeStrategy}} <: Estimator
    resampling::Union{Nothing, ResamplingStrategy}
    collaborative_strategy::S
    models::Dict
end

function CMRelevantFactorsEstimator(;
    models, 
    resampling=nothing, 
    collaborative_strategy::S=nothing
    ) where S <: Union{Nothing, CollaborativeStrategy}
    return CMRelevantFactorsEstimator{S}(resampling, collaborative_strategy, models)
end

key(estimator::CMRelevantFactorsEstimator) = 
    (CMRelevantFactorsEstimator, estimator.resampling, estimator.collaborative_strategy, estimator.models)

get_train_validation_indices(resampling, collaborative_strategy, factors, dataset) = nothing

get_train_validation_indices(resampling::ResamplingStrategy, collaborative_strategy::CollaborativeStrategy, factors, dataset) = nothing

function get_train_validation_indices(resampling::ResamplingStrategy, collaborative_strategy::Nothing, factors, dataset)
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
    collaborative_strategy = estimator.collaborative_strategy
    # Get train validation indices
    train_validation_indices = get_train_validation_indices(resampling, collaborative_strategy, estimand, dataset)
    # Fit propensity score
    propensity_score_estimator = JointConditionalDistributionEstimator(propensity_score, models, collaborative_strategy, train_validation_indices, dataset)
    propensity_score_estimate = propensity_score_estimator(
        propensity_score, 
        dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # Fit outcome mean
    outcome_model = acquire_model(models, outcome_mean.outcome, dataset, false)
    outcome_mean_estimator = ConditionalDistributionEstimator(
        outcome_model,
        train_validation_indices, 
    )
    outcome_mean_estimate = try
        outcome_mean_estimator(outcome_mean, dataset; cache=cache, verbosity=verbosity, machine_cache=machine_cache)
    catch e
        throw(FitFailedError(outcome_mean, outcome_model, outcome_mean_fit_error_msg(outcome_mean), e))
    end
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, outcome_mean_estimate, propensity_score_estimate)
    # Update cache
    cache[estimand] = estimator => estimate

    return estimate
end
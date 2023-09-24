abstract type Estimator end

#####################################################################
###               MLConditionalDistributionEstimator              ###
#####################################################################

struct MLConditionalDistributionEstimator <: Estimator
    model
end

function (estimator::MLConditionalDistributionEstimator)(estimand, dataset; cache=Dict(), verbosity=1)
    # Lookup in cache
    if haskey(cache, estimand)
        old_estimator, estimate = cache[estimand]
        if key(old_estimator) == key(estimator)
            verbosity > 0 && @info(reuse_string(estimand))
            return estimate
        end
    end
    verbosity > 0 && @info(string("Estimating: ", string_repr(estimand)))
    # Otherwise estimate
    relevant_dataset = TMLE.nomissing(dataset, TMLE.variables(estimand))
    # Fit Conditional DIstribution using MLJ
    X = selectcols(relevant_dataset, estimand.parents)
    y = Tables.getcolumn(relevant_dataset, estimand.outcome)
    mach = machine(estimator.model, X, y)
    fit!(mach, verbosity=verbosity-1)
    # Build estimate
    estimate = MLConditionalDistribution(estimand, mach)
    # Update cache
    cache[estimand] = estimator => estimate

    return estimate
end

key(estimator::MLConditionalDistributionEstimator) =
    (MLConditionalDistributionEstimator, estimator.model)

#####################################################################
###       SampleSplitMLConditionalDistributionEstimator           ###
#####################################################################

struct SampleSplitMLConditionalDistributionEstimator <: Estimator
    model
    train_validation_indices
end

function (estimator::SampleSplitMLConditionalDistributionEstimator)(estimand, dataset; cache=Dict(), verbosity=1)
    # Lookup in cache
    if haskey(cache, estimand)
        old_estimator, estimate = cache[estimand]
        if key(old_estimator) == key(estimator)
            verbosity > 0 && @info(reuse_string(estimand))
            return estimate
        end
    end
    # Otherwise estimate
    verbosity > 0 && @info(string("Estimating: ", string_repr(estimand)))
    
    relevant_dataset = TMLE.selectcols(dataset, TMLE.variables(estimand))
    nfolds = size(estimator.train_validation_indices, 1)
    machines = Vector{Machine}(undef, nfolds)
    # Fit Conditional DIstribution on each training split using MLJ
    for (index, (train_indices, _)) in enumerate(estimator.train_validation_indices)
        train_dataset = selectrows(relevant_dataset, train_indices)
        Xtrain = selectcols(train_dataset, estimand.parents)
        ytrain = Tables.getcolumn(train_dataset, estimand.outcome)
        mach = machine(estimator.model, Xtrain, ytrain)
        fit!(mach, verbosity=verbosity-1)
        machines[index] = mach
    end
    # Build estimate
    estimate = SampleSplitMLConditionalDistribution(estimand, estimator.train_validation_indices, machines)
    # Update cache
    cache[estimand] = estimator => estimate

    return estimate
end

key(estimator::SampleSplitMLConditionalDistributionEstimator) =
    (MLConditionalDistributionEstimator, estimator.model, estimator.train_validation_indices)

ConditionalDistributionEstimator(train_validation_indices::Nothing, model) =
    MLConditionalDistributionEstimator(model)

ConditionalDistributionEstimator(train_validation_indices, model) =
    SampleSplitMLConditionalDistributionEstimator(model, train_validation_indices)

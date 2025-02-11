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

struct CMRelevantFactorsEstimator <: Estimator
    resampling::Union{Nothing, ResamplingStrategy}
    models::Dict
end

CMRelevantFactorsEstimator(;models, resampling=nothing) =
    CMRelevantFactorsEstimator(resampling, models)

key(estimator::CMRelevantFactorsEstimator) = 
    (CMRelevantFactorsEstimator, estimator.resampling, estimator.models)

get_train_validation_indices(resampling::Nothing, factors, dataset) = nothing

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
    # Get train validation indices
    train_validation_indices = get_train_validation_indices(estimator.resampling, estimand, dataset)
    # Fit propensity score
    propensity_score_estimate = map(estimand.propensity_score) do factor
        try
            ps_estimator = ConditionalDistributionEstimator(train_validation_indices, acquire_model(models, factor.outcome, dataset, true))
            ps_estimator(
                factor,    
                dataset;
                cache=cache,
                verbosity=verbosity,
                machine_cache=machine_cache
            )
        catch e
            model = acquire_model(models, factor.outcome, dataset, true)
            throw(FitFailedError(factor, model, propensity_score_fit_error_msg(factor), e))
        end
    end
    # Fit outcome mean
    outcome_mean = estimand.outcome_mean
    model = acquire_model(models, outcome_mean.outcome, dataset, false)
    outcome_mean_estimator = ConditionalDistributionEstimator( 
        train_validation_indices, 
        model
    )
    outcome_mean_estimate = try
        outcome_mean_estimator(outcome_mean, dataset; cache=cache, verbosity=verbosity, machine_cache=machine_cache)
    catch e
        throw(FitFailedError(outcome_mean, model, outcome_mean_fit_error_msg(outcome_mean), e))
    end
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, outcome_mean_estimate, propensity_score_estimate)
    # Update cache
    cache[estimand] = estimator => estimate

    return estimate
end

#####################################################################
###              TargetedCMRelevantFactorsEstimator               ###
#####################################################################

struct TargetedCMRelevantFactorsEstimator
    model::Fluctuation
end

TargetedCMRelevantFactorsEstimator(Ψ, initial_factors_estimate; tol=nothing, max_iter=1, ps_lowerbound=1e-8, weighted=false, machine_cache=false) = 
    TargetedCMRelevantFactorsEstimator(Fluctuation(Ψ, initial_factors_estimate; 
        tol=tol,
        max_iter=max_iter, 
        ps_lowerbound=ps_lowerbound, 
        weighted=weighted,
        cache=machine_cache
    ))

function (estimator::TargetedCMRelevantFactorsEstimator)(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    model = estimator.model
    outcome_mean = model.initial_factors.outcome_mean.estimand
    # Fluctuate outcome model
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
    fluctuated_propensity_score = model.initial_factors.propensity_score
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, fluctuated_outcome_mean, fluctuated_propensity_score)
    # Update cache
    cache[:targeted_factors] = estimate

    return estimate
end

#####################################################################
###                            TMLE                               ###
#####################################################################

mutable struct TMLEE <: Estimator
    models::Dict
    resampling::Union{Nothing, ResamplingStrategy}
    ps_lowerbound::Union{Float64, Nothing}
    weighted::Bool
    tol::Union{Float64, Nothing}
    max_iter::Int
    machine_cache::Bool
end

"""
    TMLEE(;models=default_models(), resampling=nothing, ps_lowerbound=1e-8, weighted=false, tol=nothing, machine_cache=false)

Defines a TMLE estimator using the specified models for estimation of the nuisance parameters. The estimator is a 
function that can be applied to estimate estimands for a dataset.

# Arguments

- models (default: `default_models()`): A Dict(variable => model, ...) where the `variables` are the outcome variables modeled by the `models`.
- resampling (default: nothing): Outer resampling strategy. Setting it to `nothing` (default) falls back to vanilla TMLE while 
any valid `MLJ.ResamplingStrategy` will result in CV-TMLE.
- ps_lowerbound (default: 1e-8): Lowerbound for the propensity score to avoid division by 0. The special value `nothing` will 
result in a data adaptive definition as described in [here](https://pubmed.ncbi.nlm.nih.gov/35512316/).
- weighted (default: false): Whether the fluctuation model is a classig GLM or a weighted version. The weighted fluctuation has 
been show to be more robust to positivity violation in practice.
- tol (default: nothing): Convergence threshold for the TMLE algorithm iterations. If nothing (default), 1/(sample size) will be used. See also `max_iter`.
- max_iter (default: 1): Maximum number of iterations for the TMLE algorithm.
- machine_cache (default: false): Whether MLJ.machine created during estimation should cache data.

# Example

```julia
using MLJLinearModels
tmle = TMLEE()
Ψ̂ₙ, cache = tmle(Ψ, dataset)
```
"""
TMLEE(;models=default_models(), resampling=nothing, ps_lowerbound=1e-8, weighted=false, tol=nothing, max_iter=1, machine_cache=false) = 
    TMLEE(models, resampling, ps_lowerbound, weighted, tol, max_iter, machine_cache)

function (tmle::TMLEE)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    relevant_factors = get_relevant_factors(Ψ)
    nomissing_dataset = nomissing(dataset, variables(relevant_factors))
    initial_factors_dataset = choose_initial_dataset(dataset, nomissing_dataset, tmle.resampling)
    initial_factors_estimator = CMRelevantFactorsEstimator(tmle.resampling, tmle.models)
    initial_factors_estimate = initial_factors_estimator(relevant_factors, initial_factors_dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=tmle.machine_cache
    )
    # Get propensity score truncation threshold
    n = nrows(nomissing_dataset)
    ps_lowerbound = ps_lower_bound(n, tmle.ps_lowerbound)
    # Fluctuation initial factors
    verbosity >= 1 && @info "Performing TMLE..."
    targeted_factors_estimator = TargetedCMRelevantFactorsEstimator(
        Ψ, 
        initial_factors_estimate; 
        tol=tmle.tol,
        max_iter=tmle.max_iter,
        ps_lowerbound=ps_lowerbound, 
        weighted=tmle.weighted,
        machine_cache=tmle.machine_cache
    )
    targeted_factors_estimate = targeted_factors_estimator(relevant_factors, nomissing_dataset; 
        cache=cache, 
        verbosity=verbosity,
        machine_cache=tmle.machine_cache
        )
    # Estimation results after TMLE
    estimation_report = report(targeted_factors_estimate)

    IC = last(estimation_report.gradients)
    Ψ̂ = last(estimation_report.estimates)
    σ̂ = std(IC)
    n = size(IC, 1)
    verbosity >= 1 && @info "Done."
    return TMLEstimate(Ψ, Ψ̂, σ̂, n, IC), cache
end

gradient_and_estimate(::TMLEE, Ψ, factors, dataset; ps_lowerbound=1e-8) = 
    gradient_and_plugin_estimate(Ψ, factors, dataset; ps_lowerbound=ps_lowerbound)

#####################################################################
###                            OSE                                ###
#####################################################################

mutable struct OSE <: Estimator
    models::Dict
    resampling::Union{Nothing, ResamplingStrategy}
    ps_lowerbound::Union{Float64, Nothing}
    machine_cache::Bool
end

"""
    OSE(;models=default_models(), resampling=nothing, ps_lowerbound=1e-8, machine_cache=false)

Defines a One Step Estimator using the specified models for estimation of the nuisance parameters. The estimator is a 
function that can be applied to estimate estimands for a dataset.

# Arguments

- models: A Dict(variable => model, ...) where the `variables` are the outcome variables modeled by the `models`.
- resampling: Outer resampling strategy. Setting it to `nothing` (default) falls back to vanilla estimation while 
any valid `MLJ.ResamplingStrategy` will result in CV-OSE.
- ps_lowerbound: Lowerbound for the propensity score to avoid division by 0. The special value `nothing` will 
result in a data adaptive definition as described in [here](https://pubmed.ncbi.nlm.nih.gov/35512316/).
- machine_cache: Whether MLJ.machine created during estimation should cache data.

# Example

```julia
using MLJLinearModels
models = Dict(:Y => LinearRegressor(), :T => LogisticClassifier())
ose = OSE()
Ψ̂ₙ, cache = ose(Ψ, dataset)
```
"""
OSE(;models=default_models(), resampling=nothing, ps_lowerbound=1e-8, machine_cache=false) = 
    OSE(models, resampling, ps_lowerbound, machine_cache)

function (ose::OSE)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    initial_factors = get_relevant_factors(Ψ)
    nomissing_dataset = nomissing(dataset, variables(initial_factors))
    initial_factors_dataset = choose_initial_dataset(dataset, nomissing_dataset, ose.resampling)
    initial_factors_estimator = CMRelevantFactorsEstimator(ose.resampling, ose.models)
    initial_factors_estimate = initial_factors_estimator(
        initial_factors, 
        initial_factors_dataset;
        cache=cache, 
        verbosity=verbosity
    )
    # Get propensity score truncation threshold
    n = nrows(nomissing_dataset)
    ps_lowerbound = ps_lower_bound(n, ose.ps_lowerbound)

    # Gradient and estimate
    IC, Ψ̂ = gradient_and_estimate(ose, Ψ, initial_factors_estimate, nomissing_dataset; ps_lowerbound=ps_lowerbound)
    σ̂ = std(IC)
    n = size(IC, 1)
    verbosity >= 1 && @info "Done."
    return OSEstimate(Ψ, Ψ̂, σ̂, n, IC), cache
end

function gradient_and_estimate(::OSE, Ψ, factors, dataset; ps_lowerbound=1e-8)
    IC, Ψ̂ = gradient_and_plugin_estimate(Ψ, factors, dataset; ps_lowerbound=ps_lowerbound)
    IC_mean = mean(IC)
    IC .-= IC_mean
    return IC, Ψ̂ + IC_mean
end

#####################################################################
###                           NAIVE                               ###
#####################################################################

mutable struct NAIVE <: Estimator
    model::MLJBase.Supervised
end

function (estimator::NAIVE)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    relevant_factors = get_relevant_factors(Ψ)
    nomissing_dataset = nomissing(dataset, variables(relevant_factors))
    outcome_mean_estimate = MLConditionalDistributionEstimator(estimator.model)(
        relevant_factors.outcome_mean, 
        dataset;
        cache=cache,
        verbosity=verbosity
    )
    Ψ̂ = mean(counterfactual_aggregate(Ψ, outcome_mean_estimate, nomissing_dataset))
    return Ψ̂, cache
end

#####################################################################
###                Causal Estimand Estimation                     ###
#####################################################################


function (estimator::Union{NAIVE, OSE, TMLEE})(causalΨ::CausalCMCompositeEstimands, scm, dataset;
    identification_method=BackdoorAdjustment(),
    cache=Dict(), 
    verbosity=1
    )
    Ψ = identify(identification_method, causalΨ, scm)
    return estimator(Ψ, dataset; cache=cache, verbosity=verbosity)
end
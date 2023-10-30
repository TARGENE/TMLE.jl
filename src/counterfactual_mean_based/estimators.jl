#####################################################################
###                  CMRelevantFactorsEstimator                   ###
#####################################################################

struct CMRelevantFactorsEstimator <: Estimator
    resampling::Union{Nothing, ResamplingStrategy}
    models::NamedTuple
end

CMRelevantFactorsEstimator(;models, resampling=nothing) =
    CMRelevantFactorsEstimator(resampling, models)

key(estimator::CMRelevantFactorsEstimator) = 
    (CMRelevantFactorsEstimator, estimator.resampling, estimator.models)

get_train_validation_indices(resampling::Nothing, factors, dataset) = nothing

function get_train_validation_indices(resampling::ResamplingStrategy, factors, dataset)
    relevant_columns = collect(TMLE.variables(factors))
    outcome_variable = factors.outcome_mean.outcome
    feature_variables = filter(x -> x !== outcome_variable, relevant_columns)
    return Tuple(MLJBase.train_test_pairs(
        resampling,
        1:nrows(dataset),
        selectcols(dataset, feature_variables), 
        Tables.getcolumn(dataset, outcome_variable)
    ))
end

function (estimator::CMRelevantFactorsEstimator)(estimand, dataset; cache=Dict(), verbosity=1)
    if haskey(cache, estimand)
        old_estimator, estimate = cache[estimand]
        if key(old_estimator) == key(estimator)
            verbosity > 0 && @info(reuse_string(estimand))
            return estimate
        end
    end
    verbosity > 0 && @info(fit_string(estimand))
    models = estimator.models
    # Get train validation indices
    train_validation_indices = TMLE.get_train_validation_indices(estimator.resampling, estimand, dataset)
    # Fit propensity score
    propensity_score_estimate = Tuple(
        ConditionalDistributionEstimator(train_validation_indices, models[factor.outcome])(
            factor,    
            dataset;
            cache=cache,
            verbosity=verbosity
        ) 
        for factor in estimand.propensity_score
    )
    # Fit outcome mean
    outcome_mean = estimand.outcome_mean
    model = models[outcome_mean.outcome]
    outcome_mean_estimate = TMLE.ConditionalDistributionEstimator( 
        train_validation_indices, 
        model
        )(outcome_mean, dataset; cache=cache, verbosity=verbosity)
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

TargetedCMRelevantFactorsEstimator(Ψ, initial_factors_estimate; tol=nothing, ps_lowerbound=1e-8, weighted=false) = 
    TargetedCMRelevantFactorsEstimator(TMLE.Fluctuation(Ψ, initial_factors_estimate; 
        tol=tol, 
        ps_lowerbound=ps_lowerbound, 
        weighted=weighted
    ))

function (estimator::TargetedCMRelevantFactorsEstimator)(estimand, dataset; cache=Dict(), verbosity=1)
    model = estimator.model
    # Fluctuate outcome model
    fluctuated_outcome_mean = MLConditionalDistributionEstimator(model)(
        model.initial_factors.outcome_mean.estimand,
        dataset,
        verbosity=verbosity
    )
    # Do not fluctuate propensity score
    fluctuated_propensity_score = model.initial_factors.propensity_score
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, fluctuated_outcome_mean, fluctuated_propensity_score)
    # Update cache
    cache[:last_fluctuation] = estimate

    return estimate
end

#####################################################################
###                            TMLE                               ###
#####################################################################

mutable struct TMLEE <: Estimator
    models::NamedTuple
    resampling::Union{Nothing, ResamplingStrategy}
    ps_lowerbound::Union{Float64, Nothing}
    weighted::Bool
    tol::Union{Float64, Nothing}
end

"""
    TMLEE(models; resampling=nothing, ps_lowerbound=1e-8, weighted=false, tol=nothing)

Defines a TMLE estimator using the specified models for estimation of the nuisance parameters. The estimator is a 
function that can be applied to estimate estimands for a dataset.

# Arguments

- models: A NamedTuple{variables}(models) where the `variables` are the outcome variables modeled by the `models`.
- resampling: Outer resampling strategy. Setting it to `nothing` (default) falls back to vanilla TMLE while 
any valid `MLJ.ResamplingStrategy` will result in CV-TMLE.
- ps_lowerbound: Lowerbound for the propensity score to avoid division by 0. The special value `nothing` will 
result in a data adaptive definition as described in [here](https://pubmed.ncbi.nlm.nih.gov/35512316/).
- weighted: Whether the fluctuation model is a classig GLM or a weighted version. The weighted fluctuation has 
been show to be more robust to positivity violation in practice.
- tol: This is not used at the moment.

# Example

```julia
using MLJLinearModels
models = (Y = LinearRegressor(), T = LogisticClassifier())
tmle = TMLEE(models)
Ψ̂ₙ, cache = tmle(Ψ, dataset)
```
"""
TMLEE(models; resampling=nothing, ps_lowerbound=1e-8, weighted=false, tol=nothing) = 
    TMLEE(models, resampling, ps_lowerbound, weighted, tol)

function (tmle::TMLEE)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
    # Check the estimand against the dataset
    TMLE.check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    relevant_factors = TMLE.get_relevant_factors(Ψ)
    nomissing_dataset = TMLE.nomissing(dataset, TMLE.variables(relevant_factors))
    initial_factors_dataset = TMLE.choose_initial_dataset(dataset, nomissing_dataset, tmle.resampling)
    initial_factors_estimator = TMLE.CMRelevantFactorsEstimator(tmle.resampling, tmle.models)
    initial_factors_estimate = initial_factors_estimator(relevant_factors, initial_factors_dataset; cache=cache, verbosity=verbosity)
    # Get propensity score truncation threshold
    n = nrows(nomissing_dataset)
    ps_lowerbound = TMLE.ps_lower_bound(n, tmle.ps_lowerbound)
    # Fluctuation initial factors
    verbosity >= 1 && @info "Performing TMLE..."
    targeted_factors_estimator = TMLE.TargetedCMRelevantFactorsEstimator(
        Ψ, 
        initial_factors_estimate; 
        tol=tmle.tol, 
        ps_lowerbound=tmle.ps_lowerbound, 
        weighted=tmle.weighted
    )
    targeted_factors_estimate = targeted_factors_estimator(relevant_factors, nomissing_dataset; cache=cache, verbosity=verbosity)
    # Estimation results after TMLE
    IC, Ψ̂ = TMLE.gradient_and_estimate(Ψ, targeted_factors_estimate, nomissing_dataset; ps_lowerbound=ps_lowerbound)
    verbosity >= 1 && @info "Done."
    # update!(cache, relevant_factors, targeted_factors_estimate)
    return TMLEstimate(Ψ, Ψ̂, IC), cache
end

#####################################################################
###                            OSE                                ###
#####################################################################

mutable struct OSE <: Estimator
    models::NamedTuple
    resampling::Union{Nothing, ResamplingStrategy}
    ps_lowerbound::Union{Float64, Nothing}
end

"""
    OSE(models; resampling=nothing, ps_lowerbound=1e-8)

Defines a One Step Estimator using the specified models for estimation of the nuisance parameters. The estimator is a 
function that can be applied to estimate estimands for a dataset.

# Arguments

- models: A NamedTuple{variables}(models) where the `variables` are the outcome variables modeled by the `models`.
- resampling: Outer resampling strategy. Setting it to `nothing` (default) falls back to vanilla estimation while 
any valid `MLJ.ResamplingStrategy` will result in CV-OSE.
- ps_lowerbound: Lowerbound for the propensity score to avoid division by 0. The special value `nothing` will 
result in a data adaptive definition as described in [here](https://pubmed.ncbi.nlm.nih.gov/35512316/).

# Example

```julia
using MLJLinearModels
models = (Y = LinearRegressor(), T = LogisticClassifier())
ose = OSE(models)
Ψ̂ₙ, cache = ose(Ψ, dataset)
```
"""
OSE(models; resampling=nothing, ps_lowerbound=1e-8) = 
    OSE(models, resampling, ps_lowerbound)

function (estimator::OSE)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
    # Check the estimand against the dataset
    TMLE.check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    initial_factors = TMLE.get_relevant_factors(Ψ)
    nomissing_dataset = TMLE.nomissing(dataset, TMLE.variables(initial_factors))
    initial_factors_dataset = TMLE.choose_initial_dataset(dataset, nomissing_dataset, estimator.resampling)
    initial_factors_estimator = TMLE.CMRelevantFactorsEstimator(estimator.resampling, estimator.models)
    initial_factors_estimate = initial_factors_estimator(
        initial_factors, 
        initial_factors_dataset;
        cache=cache, 
        verbosity=verbosity
    )
    # Get propensity score truncation threshold
    n = nrows(nomissing_dataset)
    ps_lowerbound = TMLE.ps_lower_bound(n, estimator.ps_lowerbound)

    # Gradient and estimate
    IC, Ψ̂ = TMLE.gradient_and_estimate(Ψ, initial_factors_estimate, nomissing_dataset; ps_lowerbound=ps_lowerbound)
    verbosity >= 1 && @info "Done."
    return OSEstimate(Ψ, Ψ̂ + mean(IC), IC), cache
end

#####################################################################
###                           NAIVE                               ###
#####################################################################

mutable struct NAIVE <: Estimator
    model::MLJBase.Supervised
end

function (estimator::NAIVE)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
    # Check the estimand against the dataset
    TMLE.check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    relevant_factors = TMLE.get_relevant_factors(Ψ)
    nomissing_dataset = TMLE.nomissing(dataset, TMLE.variables(relevant_factors))
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
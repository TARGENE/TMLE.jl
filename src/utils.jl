###############################################################################
## General Utilities
###############################################################################
const LOCK = ReentrantLock() 

is_fluctuation_estimate(estimate::MLConditionalDistribution) = estimate.machine.model isa Fluctuation

is_fluctuation_estimate(estimate) = false

function update_cache!(cache, estimand, estimator, estimate)
    is_fluctuation_estimate(estimate) && return
    lock(LOCK) do 
        estimand_cache = get!(cache, estimand, Dict())
        estimand_cache[estimator] = estimate
    end
end

function estimate_from_cache(cache, estimand, estimator; verbosity=1)
    estimand_cache = get(cache, estimand, nothing)
    estimand_cache === nothing && return nothing
    estimate = get(estimand_cache, estimator, nothing)
    verbosity > 0 && estimate !== nothing && @info(reuse_string(estimand))
    return estimate
end

reuse_string(estimand) = string("Reusing estimate for: ", string_repr(estimand))
fit_string(estimand) = string("Estimating: ", string_repr(estimand))

unique_sorted_tuple(iter) = Tuple(sort(unique(Symbol(x) for x in iter)))

"""
For "vanilla" estimators, missingness management is deferred to the nuisance function estimators. 
This is in order to maximize data usage.
"""
choose_initial_dataset(dataset, nomissing_dataset, train_validation_indices::Nothing) = dataset

"""
For cross-validated estimators, missing data are removed early on based on all columns relevant to the estimand. 
This is to avoid the complications of:
    - Equally distributing missing across folds
    - Tracking sample_ids
"""
choose_initial_dataset(dataset, nomissing_dataset, train_validation_indices) = nomissing_dataset

"""
If no columns are provided, we return a single intercept column to accomodate marginal distribution fitting
Otherwise we return the required columns avoiding copying by default.
"""
function selectcols(dataset, colnames; copycols=false)
    return isempty(colnames) ? 
        DataFrame(INTERCEPT=ones(nrows(dataset))) : 
        DataFrames.select(dataset, collect(colnames), copycols=copycols)
end

function logit!(v)
    for i in eachindex(v)
        v[i] = logit(v[i])
    end
end

ismissingtype(T) = nonmissingtype(T) !== T

function nomissing(dataset::DataFrame, colnames; disallowmissing=true, view=false, copycols=false)
    subdataset = TMLE.selectcols(dataset, colnames, copycols=copycols)
    return if all(!ismissingtype(eltype(c)) for c in eachcol(subdataset))
        subdataset
    else
        dropmissing(subdataset, disallowmissing=disallowmissing, view=view)
    end
end

function indicator_values(indicators, T)
    indic = zeros(Float64, nrows(T))
    for (index, row) in enumerate(Tables.namedtupleiterator(T))
        indic[index] = get(indicators, values(row), 0.)
    end
    return indic
end

expected_value(ŷ::AbstractArray{<:UnivariateFinite{<:Union{OrderedFactor{2}, Multiclass{2}}}}) = pdf.(ŷ, levels(first(ŷ))[2])
expected_value(ŷ::AbstractVector{<:Distributions.UnivariateDistribution}) = mean.(ŷ)
expected_value(ŷ::AbstractVector{<:Real}) = ŷ

function counterfactualTreatment(vals, Ts)
    n = nrows(Ts)
    counterfactual_Ts = map(enumerate(names(Ts))) do (i, T_name)
        T = Ts[!, T_name]
        categorical(fill(vals[i], n), 
            levels=levels(T), 
            ordered=isordered(T)
        )
    end
    DataFrame(counterfactual_Ts, names(Ts))
    return DataFrame(counterfactual_Ts, names(Ts))
end

"""
    default_models(;Q_binary=LinearBinaryClassifier(), Q_continuous=LinearRegressor(), G=LinearBinaryClassifier()) = (

Create a Dictionary containing default models to be used by downstream estimators. 
Each provided model is prepended (in a `MLJ.Pipeline`) with an `MLJ.ContinuousEncoder`.

By default:
    - Q_binary is a LinearBinaryClassifier
    - Q_continuous is a LinearRegressor
    - G is a LinearBinaryClassifier

# Example

The following changes the default `Q_binary` to a `LogisticClassifier` and provides a `RidgeRegressor` for `special_y`. 

```julia
using MLJLinearModels
models = default_models(
    Q_binary  = LogisticClassifier(),
    special_y = RidgeRegressor()
)
```

"""
default_models(;Q_binary=LinearBinaryClassifier(), Q_continuous=LinearRegressor(), G=LinearBinaryClassifier(), kwargs...) = Dict(
    :Q_binary_default     => with_encoder(Q_binary),
    :Q_continuous_default => with_encoder(Q_continuous),
    :G_default            => with_encoder(G),
    (key => with_encoder(val) for (key, val) in kwargs)...
)

supervised_learner_supports_weights(learner) = 
    MLJBase.supports_weights(learner)

supervised_learner_supports_weights(learner::MLJBase.SupervisedPipeline) =
    MLJBase.supports_weights(MLJBase.supervised_component(learner))

is_binary(dataset, columnname) = Set(skipmissing(dataset[!, columnname])) == Set([0, 1])

function satisfies_positivity(Ψ, freq_table; positivity_constraint=0.01)
    for jointlevel in joint_levels(Ψ)
        if !haskey(freq_table, jointlevel) || freq_table[jointlevel] < positivity_constraint
            return false
        end
    end
    return true
end

satisfies_positivity(Ψ, freq_table::Nothing; positivity_constraint=nothing) = true

get_frequency_table(positivity_constraint::Nothing, dataset::Nothing, colnames) = nothing

get_frequency_table(positivity_constraint::Nothing, dataset, colnames) = nothing

get_frequency_table(positivity_constraint, dataset::Nothing, colnames) = 
    throw(ArgumentError("A dataset should be provided to enforce a positivity constraint."))

get_frequency_table(positivity_constraint, dataset, colnames) = get_frequency_table(dataset, colnames)

function get_frequency_table(dataset, colnames)
    n = nrows(dataset)
    sorted_colnames = sort(collect(colnames))
    return Dict(
        values(groupkey) => nrows(group) / n 
        for (groupkey, group) in pairs(groupby(dataset, sorted_colnames))
    )
end

function try_fit_ml_estimator(ml_estimator, conditional_distribution, dataset;
    error_fn=outcome_mean_fit_error_msg,
    cache=Dict(),
    verbosity=1,
    machine_cache=false,
    acceleration=CPU1()
    )
    return try
        ml_estimator(conditional_distribution, dataset; 
            cache=cache, 
            verbosity=verbosity, 
            machine_cache=machine_cache,
            acceleration=acceleration
            )
    catch e
        throw(FitFailedError(conditional_distribution, error_fn(conditional_distribution), e))
    end
end


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

with_encoder(model; encoder=ContinuousEncoder(drop_last=true, one_hot_ordered_factors = false)) = Pipeline(encoder,  model)

"""
    check_inputs(Ψ, dataset, prevalence)

Evaluate if the dataset is suitable for the estimand Ψ.
"""
function check_inputs(Ψ, dataset, prevalence)
    check_treatment_levels(Ψ, dataset)
    !isnothing(prevalence) && ccw_check(dataset, Ψ.outcome)
end

"""
    ccw_check(dataset, outcome)

Check if the dataset is suitable for prevalence correction (CCW-TMLE) throws an error if the outcome column is not binary or if the number of controls is lower than the number of cases.
"""
function ccw_check(dataset, outcome)
    nomissing_y = collect(skipmissing(dataset[!, outcome]))
    unique_ys = Set(nomissing_y)
    unique_ys == Set([0, 1]) || 
        throw(ArgumentError("Outcome column must be binary when prevalence is specified."))
    counts = [count(==(element), nomissing_y) for element in [0, 1]]
    counts[1] >= counts[2] || throw(ArgumentError("The dataset must contain more controls (0) than cases (1) when prevalence is provided."))
end

weighted_mean(x, w) = sum(w .* x) / sum(w)

###############################################################################
##                           Printing Utilities                             ###
###############################################################################

pretty_pvalue(pvalue) = pvalue == 0 ? "< 1e-99" : @sprintf("%.2e", pvalue)
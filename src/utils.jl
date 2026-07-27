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

censoring_indicator_name(outcome::Symbol) = Symbol(:Δ_, outcome)

has_missing_outcomes(dataset, outcome::Symbol) = ismissingtype(eltype(dataset[!, outcome]))

function add_censoring_indicator(dataset, outcome::Symbol)
    col = dataset[!, outcome]
    f(col) = categorical(ifelse.(ismissing.(col), 0, 1), ordered=true)
    DataFrames.transform(dataset, outcome => f => censoring_indicator_name(outcome))
end

function compute_ipcw_weights(censoring_score, dataset; ps_lowerbound=1e-8)
    censoring_score === nothing && return nothing
    outcome = selectcols(dataset, [censoring_score.estimand.outcome])
    Δ = indicator_values(Dict((1,) => 1.), outcome)
    π = likelihood(censoring_score, dataset)
    truncate!(π, ps_lowerbound)
    return Δ ./ π
end

"""
    counterfactual_censoring_covariate(censoring_score, dataset; ps_lowerbound=1e-8)

Censoring factor for the counterfactual censoring intervention `Δ=1`: `1/P(Δ=1 | parents)`.
Unlike `compute_ipcw_weights` (which uses the observed `Δ`), this sets `Δ=1`, so it is the
factor folded into the *counterfactual* clever covariate.
"""
function counterfactual_censoring_covariate(censoring_score, dataset; ps_lowerbound=1e-8)
    π = expected_value(censoring_score, dataset)
    truncate!(π, ps_lowerbound)
    return 1 ./ π
end

function nomissing(dataset::DataFrame, colnames; disallowmissing=true, view=false, copycols=false)
    subdataset = TMLE.selectcols(dataset, colnames, copycols=copycols)
    return if all(!ismissingtype(eltype(c)) for c in eachcol(subdataset))
        subdataset
    else
        dropmissing(subdataset, disallowmissing=disallowmissing, view=view)
    end
end


"""
    get_initial_dataset(dataset, relevant_factors; prevalence=nothing, verbosity=1)

Build the single dataset used for both fold construction and nuisance fitting. Keeping these on
the same rows means the CV fold indices stay aligned with the fluctuation dataset (which is these
same rows with missing outcomes coalesced, see `get_fluctuation_dataset`).

- **IPCW mode** (`relevant_factors.censoring_score !== nothing`): keeps every row. Each nuisance
  estimator drops the rows it can't use (missing among *its own* variables) when it fits, so the
  propensity/censoring models are not penalised by rows another model happens to be missing. The
  covariate-complete rows needed to predict *all* nuisances are selected separately by
  `get_fluctuation_dataset`; under CV the fold indices are realigned to that subset via
  `align_to_rows` in the estimators.
- **Non-IPCW mode**: drops all rows with any missing relevant variable. If `prevalence` is
  provided, additionally applies matched-controls subsampling via `get_matched_controls`.
"""
function get_initial_dataset(dataset, relevant_factors; prevalence=nothing, verbosity=1)
    relevant_factors.censoring_score !== nothing && return dataset
    outcome = relevant_factors.outcome_mean.outcome
    nomissing_dataset = nomissing(dataset, variables(relevant_factors))
    return isnothing(prevalence) ? nomissing_dataset : get_matched_controls(nomissing_dataset, outcome; verbosity=verbosity)
end

"""
    get_fluctuation_dataset(initial_dataset, relevant_factors)

Derive the dataset on which all nuisances are predicted to fit the fluctuation (epsilon) and
evaluate the gradient. Under IPCW it (1) drops rows missing any covariate — every nuisance
(Q, G, π) must be predictable on every row, which coalescing the outcome alone does not
guarantee — and (2) coalesces missing outcomes to 0 so the fluctuation GLM has a numeric target;
censored rows (Δ=0) are zeroed out by the clever covariate, so the imputed value is irrelevant.
Outside IPCW the initial dataset is already complete and is returned unchanged.

The kept rows are those flagged by `fluctuation_row_mask`; CV fold indices are realigned to this
same subset via `align_to_rows`, so the two stay consistent.
"""
function get_fluctuation_dataset(initial_dataset, relevant_factors)
    relevant_factors.censoring_score === nothing && return initial_dataset
    outcome = relevant_factors.outcome_mean.outcome
    fluctuation_dataset = initial_dataset[fluctuation_row_mask(initial_dataset, relevant_factors), :]
    fluctuation_dataset[!, outcome] = coalesce_outcome(fluctuation_dataset[!, outcome])
    return fluctuation_dataset
end

"""
    fluctuation_row_mask(initial_dataset, relevant_factors)

Boolean mask (over the full initial-dataset rows) selecting the covariate-complete rows that make
up the fluctuation dataset — every nuisance must be predictable on these rows. Single source of
truth shared by `get_fluctuation_dataset` and the CV fold realignment in the estimators.
"""
function fluctuation_row_mask(initial_dataset, relevant_factors)
    outcome = relevant_factors.outcome_mean.outcome
    covariate_vars = filter(!=(outcome), collect(variables(relevant_factors)))
    return completecases(initial_dataset, covariate_vars)
end

"""
    coalesce_outcome(y)

Replace missing outcome values with a placeholder (first level for a categorical, `zero` for a
numeric) and return a column with a non-missing element type.
"""
function coalesce_outcome(y)
    ismissingtype(eltype(y)) || return y
    if y isa CategoricalVector
        lvls = levels(y)
        raw = [ismissing(v) ? lvls[1] : unwrap(v) for v in y]
        return categorical(raw, levels=lvls, ordered=isordered(y))
    end
    return coalesce.(y, zero(nonmissingtype(eltype(y))))
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
    return DataFrame(counterfactual_Ts, names(Ts))
end

"""
    get_matched_controls(dataset, relevant_factors, J)

Returns the matched controls for each case in the dataset based on the intended number of controls per case (J).
Randomly discards unmatched controls.
Currently, this implementation is for independent case-control studies. Will be expanded for matched case-control studies in the future.
"""
function get_matched_controls(dataset, outcome; verbosity = 1)
    y = dataset[!, outcome]
    idx_case = findall(y .== 1)
    idx_ctl  = findall(y .== 0)
    nC  = length(idx_case)
    nCo = length(idx_ctl)
    J, surplus = divrem(nCo, nC)
    if surplus !== 0
        verbosity > 0 && @info("Dropping $surplus control(s) to ensure equal number of controls per case (J=$J). You can pre-drop these controls yourself to prevent this operation.")
        samples_to_drop = shuffle!(idx_ctl)[1:surplus]
        return dataset[Not(samples_to_drop), :]
    else
        return dataset
    end
end

"""
    default_models(;Q_binary=LinearBinaryClassifier(), Q_continuous=LinearRegressor(), G=LinearBinaryClassifier(), C=LinearBinaryClassifier())

Create a Dictionary containing default models to be used by downstream estimators. 
Each provided model is prepended (in a `MLJ.Pipeline`) with an `MLJ.ContinuousEncoder`.

By default:
    - Q_binary is a LinearBinaryClassifier (binary outcome mean)
    - Q_continuous is a LinearRegressor (continuous outcome mean)
    - G is a LinearBinaryClassifier (propensity score)
    - C is a LinearBinaryClassifier (censoring score for IPCW)

The `C` model is used when outcome missingness triggers IPCW (see `Tmle` / `Ose`). It models
`P(Δ=1 | W)`, the probability of observing the outcome given covariates. You can also assign
a model to the censoring indicator name directly (e.g. `Symbol("Δ_Y") => your_model`).

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
default_models(;Q_binary=LinearBinaryClassifier(), Q_continuous=LinearRegressor(), G=LinearBinaryClassifier(), C=LinearBinaryClassifier(), kwargs...) = Dict(
    :Q_binary_default     => with_encoder(Q_binary),
    :Q_continuous_default => with_encoder(Q_continuous),
    :G_default            => with_encoder(G),
    :C_default            => with_encoder(C),
    (key => with_encoder(val) for (key, val) in kwargs)...
)

supervised_learner_supports_weights(learner) = 
    MLJBase.supports_weights(get_predictor(learner))

get_predictor(learner::MLJBase.SupervisedPipeline) = 
    get_predictor(MLJBase.supervised_component(learner))

get_predictor(learner::MLJBase.Supervised) = learner

get_predictor(learner) = 
    throw(ArgumentError("Only learners of type `Supervised` and `SupervisedPipeline` are supported for CCW-TMLE. $(typeof(learner)) is not."))

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

###############################################################################
##                           Printing Utilities                             ###
###############################################################################

pretty_pvalue(pvalue) = pvalue == 0 ? "< 1e-99" : @sprintf("%.2e", pvalue)
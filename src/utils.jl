###############################################################################
## General Utilities
###############################################################################

selectcols(data, cols) = data |> TableOperations.select(cols...) |> Tables.columntable

data_adaptive_ps_lower_bound(n::Int; max_lb=0.1) = 
    min(5 / (√(n)*log(n/5)), max_lb)

"""
    data_adaptive_ps_lower_bound(Ψ::CMCompositeEstimand)

This startegy is from [this paper](https://academic.oup.com/aje/article/191/9/1640/6580570?login=false) 
but the study does not show strictly better behaviour of the strategy so not a default for now.
"""
function data_adaptive_ps_lower_bound(Ψ::CMCompositeEstimand;max_lb=0.1)
    n = nrows(get_outcome_datas(Ψ)[2])
    return data_adaptive_ps_lower_bound(n; max_lb=max_lb) 
end

ps_lower_bound(Ψ::CMCompositeEstimand, lower_bound::Nothing; max_lb=0.1) = data_adaptive_ps_lower_bound(Ψ; max_lb=max_lb)
ps_lower_bound(Ψ::CMCompositeEstimand, lower_bound; max_lb=0.1) = min(max_lb, lower_bound)


function logit!(v)
    for i in eachindex(v)
        v[i] = logit(v[i])
    end
end

function truncate!(v::AbstractVector, ps_lowerbound)
    for i in eachindex(v)
        v[i] = max(v[i], ps_lowerbound)
    end
end

function nomissing(table)
    sch = Tables.schema(table)
    for type in sch.types
        if nonmissingtype(type) != type
            coltable = table |> 
                       TableOperations.dropmissing |> 
                       Tables.columntable
            return NamedTuple{keys(coltable)}([disallowmissing(col) for col in coltable])
        end
    end
    return table
end

function nomissing(table, columns)
    columns = selectcols(table, columns)
    return nomissing(columns)
end

ncases(value, Ψ::Estimand) = sum(value[i] == Ψ.treatment[i].case for i in eachindex(value))

function indicator_fns(Ψ::IATE)
    N = length(treatments(Ψ))
    key_vals = Pair[]
    for cf in Iterators.product((values(Ψ.treatment[T]) for T in treatments(Ψ))...)
        push!(key_vals, cf => float((-1)^(N - ncases(cf, Ψ))))
    end
    return Dict(key_vals...)
end

indicator_fns(Ψ::CM) = Dict(values(Ψ.treatment) => 1.)

function indicator_fns(Ψ::ATE)
    case = []
    control = []
    for treatment in Ψ.treatment
        push!(case, treatment.case)
        push!(control, treatment.control)
    end
    return Dict(Tuple(case) => 1., Tuple(control) => -1.)
end

function indicator_values(indicators, T)
    indic = zeros(Float64, nrows(T))
    for (index, row) in enumerate(Tables.namedtupleiterator(T))
        indic[index] = get(indicators, values(row), 0.)
    end
    return indic
end

NotIdentifiedError(reasons) = ArgumentError(string(
    "The estimand is not identified for the following reasons: \n\t- ", join(reasons, "\n\t- ")))

AbsentLevelError(treatment_name, key, val, levels) = ArgumentError(string(
    "The treatment variable ", treatment_name, "'s, '", key, "' level: '", val,
    "' in Ψ does not match any level in the dataset: ", levels))

AbsentLevelError(treatment_name, val, levels) = ArgumentError(string(
    "The treatment variable ", treatment_name, "'s, level: '", val,
    "' in Ψ does not match any level in the dataset: ", levels))

"""
    check_treatment_settings(settings::NamedTuple, levels, treatment_name)

Checks the case/control values defining the treatment contrast are present in the dataset levels. 

Note: This method is for estimands like the ATE or IATE that have case/control treatment settings represented as 
`NamedTuple`.
"""
function check_treatment_settings(settings::NamedTuple, levels, treatment_name)
    for (key, val) in zip(keys(settings), settings) 
        any(val .== levels) || 
            throw(AbsentLevelError(treatment_name, key, val, levels))
    end
end

"""
    check_treatment_settings(setting, levels, treatment_name)

Checks the value defining the treatment setting is present in the dataset levels. 

Note: This is for estimands like the CM that do not have case/control treatment settings 
and are represented as simple values.
"""
function check_treatment_settings(setting, levels, treatment_name)
    any(setting .== levels) || 
            throw(
                AbsentLevelError(treatment_name, setting, levels))
end

"""
    check_treatment_levels(Ψ::CMCompositeEstimand, dataset)

Makes sure the defined treatment levels are present in the dataset.
"""
function check_treatment_levels(Ψ::CMCompositeEstimand, dataset)
    for treatment_name in treatments(Ψ)
        treatment_levels = levels(Tables.getcolumn(dataset, treatment_name))
        treatment_settings = getproperty(Ψ.treatment, treatment_name)
        check_treatment_settings(treatment_settings, treatment_levels, treatment_name)
    end
end

expected_value(ŷ::UnivariateFiniteVector{Multiclass{2}}) = pdf.(ŷ, levels(first(ŷ))[2])
expected_value(ŷ::AbstractVector{<:Distributions.UnivariateDistribution}) = mean.(ŷ)
expected_value(ŷ::AbstractVector{<:Real}) = ŷ

function counterfactualTreatment(vals, T)
    Tnames = Tables.columnnames(T)
    n = nrows(T)
    NamedTuple{Tnames}(
            [categorical(repeat([vals[i]], n), levels=levels(Tables.getcolumn(T, name)), ordered=isordered(Tables.getcolumn(T, name)))
                            for (i, name) in enumerate(Tnames)])
end

function compute_offset(ŷ::UnivariateFiniteVector{Multiclass{2}})
    μy = expected_value(ŷ)
    logit!(μy)
    return μy
end
compute_offset(ŷ::AbstractVector{<:Distributions.UnivariateDistribution}) = expected_value(ŷ)
compute_offset(ŷ::AbstractVector{<:Real}) = expected_value(ŷ)

compute_offset(Ψ::CMCompositeEstimand) = 
    compute_offset(MLJBase.predict(get_outcome_model(Ψ)))

function balancing_weights(Ψ::CMCompositeEstimand, W, T; ps_lowerbound=1e-8)
    density = ones(nrows(T))
    for colname ∈ Tables.columnnames(T)
        mach = Ψ.scm[colname].mach
        ŷ = MLJBase.predict(mach, W[colname])
        density .*= pdf.(ŷ, Tables.getcolumn(T, colname))
    end
    truncate!(density, ps_lowerbound)
    return 1. ./ density
end

"""
    clever_covariate_and_weights(jointT, W, G, indicator_fns; ps_lowerbound=1e-8, weighted_fluctuation=false)

Computes the clever covariate and weights that are used to fluctuate the initial Q.

if `weighted_fluctuation = false`:

- ``clever_covariate(t, w) = \\frac{SpecialIndicator(t)}{p(t|w)}`` 
- ``weight(t, w) = 1``

if `weighted_fluctuation = true`:

- ``clever_covariate(t, w) = SpecialIndicator(t)`` 
- ``weight(t, w) = \\frac{1}{p(t|w)}``

where SpecialIndicator(t) is defined in `indicator_fns`.
"""
function clever_covariate_and_weights(Ψ::CMCompositeEstimand, X; ps_lowerbound=1e-8, weighted_fluctuation=false)
    # Compute the indicator values
    T = treatments(X, Ψ)
    W = confounders(X, Ψ)
    indic_vals = indicator_values(indicator_fns(Ψ), T)
    weights = balancing_weights(Ψ, W, T, ps_lowerbound=ps_lowerbound)
    if weighted_fluctuation
        return indic_vals, weights
    end
    # Vanilla unweighted fluctuation
    indic_vals .*= weights
    return indic_vals, ones(size(weights, 1))
end

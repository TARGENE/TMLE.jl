###############################################################################
## General Utilities
###############################################################################

selectcols(data, cols) = data |> TableOperations.select(cols...) |> Tables.columntable

function logit!(v)
    for i in eachindex(v)
        v[i] = logit(v[i])
    end
end

function plateau!(v::AbstractVector, threshold)
    for i in eachindex(v)
        v[i] = max(v[i], threshold)
    end
end

joint_name(it) = join(it, "_&_")

joint_treatment(T) =
    categorical(joint_name.(Tables.rows(T)))


log_fit(verbosity, model) = 
    verbosity >= 1 && @info string("→ Fitting ", model)

log_no_fit(verbosity, model) =
    verbosity >= 1 && @info string("→ Reusing previous ", model)

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

"""
"""
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

###############################################################################
## Offset & Covariate
###############################################################################

expected_value(ŷ::UnivariateFiniteVector{Multiclass{2}}) = pdf.(ŷ, levels(first(ŷ))[2])
expected_value(ŷ::AbstractVector{<:Distributions.UnivariateDistribution}) = mean.(ŷ)
expected_value(ŷ::AbstractVector{<:Real}) = ŷ

function compute_offset(ŷ::UnivariateFiniteVector{Multiclass{2}})
    μy = expected_value(ŷ)
    logit!(μy)
    return μy
end
compute_offset(ŷ::AbstractVector{<:Distributions.UnivariateDistribution}) = expected_value(ŷ)
compute_offset(ŷ::AbstractVector{<:Real}) = expected_value(ŷ)

compute_offset(Ψ::CMCompositeEstimand) = 
    compute_offset(MLJBase.predict(outcome_equation(Ψ).mach))

function balancing_weights(scm, W, T; threshold=1e-8)
    density = ones(nrows(T))
    for colname ∈ Tables.columnnames(T)
        mach = scm[colname].mach
        ŷ = MLJBase.predict(mach, W[colname])
        density .*= pdf.(ŷ, Tables.getcolumn(T, colname))
    end
    plateau!(density, threshold)
    return 1. ./ density
end

"""
    clever_covariate_and_weights(jointT, W, G, indicator_fns; threshold=1e-8, weighted_fluctuation=false)

Computes the clever covariate and weights that are used to fluctuate the initial Q.

if `weighted_fluctuation = false`:

- ``clever_covariate(t, w) = \\frac{SpecialIndicator(t)}{p(t|w)}`` 
- ``weight(t, w) = 1``

if `weighted_fluctuation = true`:

- ``clever_covariate(t, w) = SpecialIndicator(t)`` 
- ``weight(t, w) = \\frac{1}{p(t|w)}``

where SpecialIndicator(t) is defined in `indicator_fns`.
"""
function clever_covariate_and_weights(scm::SCM, T, W, indicator_fns; threshold=1e-8, weighted_fluctuation=false)
    # Compute the indicator values
    indic_vals = TMLE.indicator_values(indicator_fns, T)
    weights = balancing_weights(scm, W, T, threshold=threshold)
    if weighted_fluctuation
        return indic_vals, weights
    end
    # Vanilla unweighted fluctuation
    indic_vals .*= weights
    return indic_vals, ones(size(weights, 1))
end

###############################################################################
## Fluctuation
###############################################################################

fluctuation_input(covariate::AbstractVector{T}, offset::AbstractVector{T}) where T = (covariate=covariate, offset=offset)

"""

The GLM models require inputs of the same type
"""
fluctuation_input(covariate::AbstractVector{T1}, offset::AbstractVector{T2}) where {T1, T2} = 
    (covariate=covariate, offset=convert(Vector{T1}, offset))


function counterfactualTreatment(vals, T)
    Tnames = Tables.columnnames(T)
    n = nrows(T)
    NamedTuple{Tnames}(
            [categorical(repeat([vals[i]], n), levels=levels(Tables.getcolumn(T, name)), ordered=isordered(Tables.getcolumn(T, name)))
                            for (i, name) in enumerate(Tnames)])
end




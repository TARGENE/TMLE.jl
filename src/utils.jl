###############################################################################
## General Utilities
###############################################################################

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

ncases(value, Ψ::Parameter) = sum(value[i] == Ψ.treatment[i].case for i in eachindex(value))

function indicator_fns(Ψ::IATE, f::Function)
    N = length(treatments(Ψ))
    key_vals = Pair[]
    for cf in Iterators.product((values(Ψ.treatment[T]) for T in treatments(Ψ))...)
        push!(key_vals, f(cf) => float((-1)^(N - ncases(cf, Ψ))))
    end
    return Dict(key_vals...)
end

indicator_fns(Ψ::CM, f::Function) = Dict(f(values(Ψ.treatment)) => 1.)

function indicator_fns(Ψ::ATE, f::Function)
    case = []
    control = []
    for treatment in Ψ.treatment
        push!(case, treatment.case)
        push!(control, treatment.control)
    end
    return Dict(f(Tuple(case)) => 1., f(Tuple(control)) => -1.)
end

function indicator_values(indicators, jointT)
    indic = zeros(Float64, nrows(jointT))
    for i in eachindex(jointT)
        val = jointT[i]
        indic[i] = get(indicators, val, 0.)
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

function balancing_weights(G::Machine, W, jointT; threshold=0.005)
    ŷ = MLJBase.predict(G, W)
    d = pdf.(ŷ, jointT)
    plateau!(d, threshold)
    return 1. ./ d
end

function clever_covariate_and_weights(jointT, W, G, indicator_fns; threshold=0.005, weighted_fluctuation=false)
    # Compute the indicator values
    indic_vals = TMLE.indicator_values(indicator_fns, jointT)
    weights = balancing_weights(G, W, jointT, threshold=threshold)
    if weighted_fluctuation
        return indic_vals, weights
    end
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




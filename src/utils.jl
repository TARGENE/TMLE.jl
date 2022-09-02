###############################################################################
## General Utilities
###############################################################################

logit(X) = log.(X ./ (1 .- X))
expit(X) = 1 ./ (1 .+ exp.(-X))


log_fit(verbosity, model) = 
    verbosity >= 1 && @info string("→ Fitting ", model)

log_no_fit(verbosity, model) =
    verbosity >= 1 && @info string("→ Reusing previous ", model)

"""

Adapts the type of the treatment variable passed to the G learner
"""
adapt(T) =
    size(Tables.columnnames(T), 1) == 1 ? Tables.getcolumn(T, 1) : T

low_propensity_scores(Gmach, W, T, threshold) =
    findall(<(threshold), density(Gmach, W, T))


totable(x::AbstractVector) = (y=x,)
totable(x) = x

function nomissing(table)
    sch = Tables.schema(table)
    for type in sch.types
        if nonmissingtype(type) != type
            return table |> 
                   TableOperations.dropmissing |> 
                   Tables.columntable
        end
    end
    return table
end

function nomissing(table, columns)
    columns = selectcols(table, columns)
    return nomissing(columns)
end

ncases(value, Ψ::Parameter) = sum(value[i] == Ψ.treatment[i].case for i in eachindex(value))

function indicator_fns(Ψ::IATE)
    N = length(treatments(Ψ))
    indicators = Dict()
    for cf in Iterators.product((values(Ψ.treatment[T]) for T in treatments(Ψ))...)
        indicators[cf] = (-1)^(N - ncases(cf, Ψ))
    end
    return indicators
end

indicator_fns(Ψ::CM) = Dict(values(Ψ.treatment) => 1)

function indicator_fns(Ψ::ATE) 
    case = []
    control = []
    for treatment in Ψ.treatment
        push!(case, treatment.case)
        push!(control, treatment.control)
    end
    return Dict(Tuple(case) => 1, Tuple(control) => -1)
end


###############################################################################
## Offset
###############################################################################
expected_value(ŷ, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:Finite}}) = pdf.(ŷ, levels(first(ŷ))[2])
expected_value(ŷ, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:MLJBase.Continuous}}) = mean.(ŷ)
expected_value(ŷ, ::Type{<:Deterministic}, ::Type{<:AbstractArray{<:MLJBase.Continuous}}) = ŷ


maybelogit(x, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:Finite}}) = logit(x)
maybelogit(x, _, _) = x

function compute_offset(mach::Machine, X)
    ŷ = MLJBase.predict(mach, X)
    expectation = expected_value(ŷ, typeof(mach.model), target_scitype(mach.model))
    return maybelogit(expectation, typeof(mach.model), target_scitype(mach.model))
end


###############################################################################
## Covariate
###############################################################################

function indicator_values(indicators, T)
    N = length(first(keys(indicators)))
    covariate = zeros(Float64, nrows(T))
    for (i, row) in enumerate(Tables.rows(T))
        vals = Tuple(Tables.getcolumn(row, nm) for nm in 1:N)
        if haskey(indicators, vals)
            covariate[i] = indicators[vals]
        end
    end
    covariate
end


plateau_likelihood(likelihood, threshold) = max.(threshold, likelihood)


"""
For each data point, computes: (-1)^(interaction-oder - j)
Where j is the number of treatments different from the reference in the query.
"""
function compute_covariate(Gmach::Machine, W, T, indicators; threshold=0.005)
    # Compute the indicator value
    indic_vals = TMLE.indicator_values(indicators, T)

    # Compute density and truncate
    likelihood = TMLE.density(Gmach, W, T)

    likelihood = plateau_likelihood(likelihood, threshold)
    
    return indic_vals ./ likelihood
end


###############################################################################
## Fluctuation
###############################################################################

influencecurve(covariate, y, observed_fluct, ct_fluct, estimate) = 
    covariate .* (float(y) .- observed_fluct) .+ ct_fluct .- estimate

fluctuation_input(covariate, offset) = (covariate=covariate, offset=offset)

function counterfactualTreatment(vals, T)
    Tnames = Tables.columnnames(T)
    n = nrows(T)
    NamedTuple{Tnames}(
            [categorical(repeat([vals[i]], n), levels=levels(Tables.getcolumn(T, name)))
                            for (i, name) in enumerate(Tnames)])
end




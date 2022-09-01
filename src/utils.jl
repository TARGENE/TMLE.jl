###############################################################################
## General Utilities
###############################################################################

logit(X) = log.(X ./ (1 .- X))
expit(X) = 1 ./ (1 .+ exp.(-X))

"""

Let's default to no warnings for now.
"""
MLJBase.check(model::TMLEstimator, args... ; full=false) = true

"""

Adapts the type of the treatment variable passed to the G learner
"""
adapt(T) =
    size(Tables.columnnames(T), 1) == 1 ? Tables.getcolumn(T, 1) : T

low_propensity_scores(Gmach, W, T, threshold) =
    findall(<(threshold), density(Gmach, W, T))


totable(x::AbstractVector) = (y=x,)
totable(x) = x

function merge_and_dropmissing(tables...)
    return mapreduce(t->Tables.columntable(t), merge, tables) |> 
        TableOperations.dropmissing |> 
        Tables.columntable |>
        disallowmissings
end

function disallowmissings(T)
    newcols = AbstractVector[]
    sch = Tables.schema(T)
    Tables.eachcolumn(sch, T) do col, _, _
        push!(newcols, disallowmissing(col))
    end
    return NamedTuple{sch.names}(newcols)
end

function TableOperations.dropmissing(tables...)
    table = merge_and_dropmissing(tables...)
    table = disallowmissings(table)
    return Tuple(Tables.columntable(TableOperations.select(table, Tables.columnnames(t)...)) for t in tables)
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


function compute_fluctuation(Fmach::Machine, 
                             Q̅mach::Machine, 
                             Gmach::Machine, 
                             indicators,
                             W, 
                             T,
                             X; 
                             threshold=0.005)
    offset = compute_offset(Q̅mach, X)
    covariate = compute_covariate(Gmach, W, T, indicators; 
                                    threshold=threshold)
    Xfluct = fluctuation_input(covariate, offset)
    return predict_mean(Fmach, Xfluct)
end

###############################################################################
## Report Generation
###############################################################################

"""

For a given query, identified by `indicators`, reports the different quantities of
interest. An important intermediate quantity is obtained by aggregation of 
E[Y|T, W] evaluated at the various counterfactual values of the treatment.
For instance, if the order of Interaction is 2 with binary variables, this is computed as:

AggregatedCounterfactual = Fluctuation(t₁=1, t₂=1, W=w) - Fluctuation(t₁=1, t₂=0, W=w)
                - Fluctuation(t₁=0, t₂=1, W=w) + Fluctuation(t₁=0, t₂=0, W=w)
"""
function tmlereport(Fmach::Machine,
                    Q̅mach::Machine,
                    Gmach::Machine,
                    Hmach::Machine,
                    W,
                    T,
                    observed_fluct,
                    ys,
                    covariate,
                    indicators, 
                    threshold,
                    query,
                    target_name)

    tmle_ct_agg = zeros(nrows(T))
    initial_ct_agg = zeros(nrows(T))
    for (vals, sign) in indicators 
        counterfactualT = counterfactualTreatment(vals, T)
        Thot = transform(Hmach, counterfactualT)
        X = merge(Thot, W)

        initial_expectation = expected_value(MLJBase.predict(Q̅mach, X), typeof(Q̅mach.model), target_scitype(Q̅mach.model))
        initial_ct_agg .+= sign.*initial_expectation
        
        tmle_ct_agg .+= sign.*compute_fluctuation(Fmach, 
                    Q̅mach, 
                    Gmach,
                    indicators,
                    W, 
                    counterfactualT,
                    X; 
                    threshold=threshold)
    end

    initial_estimate = mean(initial_ct_agg)
    tmle_estimate = mean(tmle_ct_agg)
    inf_curve = influencecurve(covariate, ys, observed_fluct, tmle_ct_agg, tmle_estimate)

    return TMLEReport(target_name, query, inf_curve, tmle_estimate, initial_estimate)
end

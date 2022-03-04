###############################################################################
## General Utilities
###############################################################################

logit(X) = log.(X ./ (1 .- X))
logit(X::AbstractNode) = node(x->logit(x), X)

expit(X) = 1 ./ (1 .+ exp.(-X))
expit(X::AbstractNode) = node(x->expit(x), X)

"""

Let's default to no warnings for now.
"""
MLJBase.check(model::TMLEstimator, args... ; full=false) = true

Base.merge(ndt₁::AbstractNode, ndt₂::AbstractNode) = 
    node((ndt₁, ndt₂) -> merge(ndt₁, ndt₂), ndt₁, ndt₂)

"""

Adapts the type of the treatment variable passed to the G learner
"""
adapt(T::NamedTuple{<:Any, NTuple{1, Z}}) where Z = T[1]
adapt(T) = T
adapt(T::AbstractNode) = node(adapt, T)


function log_over_threshold(covariate::AbstractNode, threshold)
    node(cov -> findall(x -> x >= 1/threshold, cov), covariate)
end

Tables.getcolumn(T::AbstractNode, name::Symbol) = 
    node(T->Tables.getcolumn(T, name), T)

###############################################################################
## Offset
###############################################################################
expected_value(ŷ, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:Finite}}) = pdf.(ŷ, levels(first(ŷ))[2])
expected_value(ŷ::AbstractNode, t::Type{<:Probabilistic}, s::Type{<:AbstractArray{<:Finite}}) = 
    node(ŷ->expected_value(ŷ, t, s), ŷ)

expected_value(ŷ, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:MLJBase.Continuous}}) = mean.(ŷ)
expected_value(ŷ::AbstractNode, t::Type{<:Probabilistic}, s::Type{<:AbstractArray{<:MLJBase.Continuous}}) = 
    node(ŷ->expected_value(ŷ, t, s), ŷ)

expected_value(ŷ, ::Type{<:Deterministic}, ::Type{<:AbstractArray{<:MLJBase.Continuous}}) = ŷ
expected_value(ŷ::AbstractNode, t::Type{<:Deterministic}, s::Type{<:AbstractArray{<:MLJBase.Continuous}}) = 
    node(ŷ->expected_value(ŷ, t, s), ŷ)

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

function indicator_values(indicators::ImmutableDict{<:NTuple{N, Any}, ValType}, T) where {N, ValType}
    covariate = zeros(ValType, nrows(T))
    for (i, row) in enumerate(Tables.rows(T))
        vals = Tuple(Tables.getcolumn(row, nm) for nm in 1:N)
        if haskey(indicators, vals)
            covariate[i] = indicators[vals]
        end
    end
    covariate
end

function _indicator_values(indicators::ImmutableDict{<:NTuple{N, Any}, ValType}, T) where {N, ValType}
    n = nrows(T)
    T_ = hcat(T...)
    covariate = zeros(ValType, n)
    for i in 1:n
        vals = view(T_, i, :)
        if haskey(indicators, vals)
            covariate[i] = indicators[vals]
        end
    end
    covariate
end


indicator_values(indicators::ImmutableDict{<:NTuple{N, Any}, ValType}, T::AbstractNode) where {N, ValType} = 
    node(t -> indicator_values(indicators, t), T)


plateau_likelihood(likelihood, threshold) = max.(threshold, likelihood)
plateau_likelihood(likelihood::AbstractNode, threshold) = 
    node(l -> plateau_likelihood(l, threshold), likelihood)

elemwise_divide(x, y) = x ./ y
elemwise_divide(x::AbstractNode, y::AbstractNode) = 
    node((x, y) -> elemwise_divide(x,y), x, y)

"""
For each data point, computes: (-1)^(interaction-oder - j)
Where j is the number of treatments different from the reference in the query.
"""
function compute_covariate(Gmach::Machine, W, T, indicators; threshold=0.005)
    # Compute the indicator value
    indic_vals = indicator_values(indicators, T)

    # Compute density and truncate
    likelihood = density(Gmach, W, T)

    likelihood = plateau_likelihood(likelihood, threshold)
    
    return elemwise_divide(indic_vals, likelihood)
end


###############################################################################
## Fluctuation
###############################################################################

fluctuation_input(covariate, offset) = (covariate=covariate, offset=offset)
fluctuation_input(covariate::AbstractNode, offset::AbstractNode) =
    node((c, o) -> fluctuation_input(c, o), covariate, offset)
    
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

function estimation_report(Fmach::Machine,
    Q̅mach::Machine,
    Gmach::Machine,
    Hmach::Machine,
    W::AbstractNode,
    T::AbstractNode,
    observed_fluct::AbstractNode,
    ys::AbstractNode,
    covariate::AbstractNode,
    indicators,
    threshold,
    query)

    node((w, t, o, y, c) -> estimation_report(Fmach, Q̅mach, Gmach, Hmach, w, t, o, y, c, indicators, threshold, query), 
                                W, T, observed_fluct, ys, covariate)
end

"""

For a given query, identified by `indicators`, reports the different quantities of
interest. An important intermediate quantity is obtained by aggregation of 
E[Y|T, W] evaluated at the various counterfactual values of the treatment.
For instance, if the order of Interaction is 2 with binary variables, this is computed as:

AggregatedCounterfactual = Fluctuation(t₁=1, t₂=1, W=w) - Fluctuation(t₁=1, t₂=0, W=w)
                - Fluctuation(t₁=0, t₂=1, W=w) + Fluctuation(t₁=0, t₂=0, W=w)
"""
function estimation_report(Fmach::Machine,
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
                            query)

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

    return QueryReport(query, inf_curve, tmle_estimate, initial_estimate)
end

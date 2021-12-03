###############################################################################
## General Utilities
###############################################################################

logit(X) = log.(X ./ (1 .- X))
logit(X::AbstractNode) = node(x->logit(x), X)

expit(X) = 1 ./ (1 .+ exp.(-X))
expit(X::AbstractNode) = node(x->expit(x), X)


"""
Hack into GLM to compute deviance on y a real
"""
function GLM.devresid(::Bernoulli, y::Vector{<:Real}, μ::Real)
    return -2*(y*log(μ) + (1-y)*log1p(-μ))
end

"""
Remove default check for y to be binary
"""
GLM.checky(y, d::Bernoulli) = nothing

"""

Let's default to no warnings for now.
"""
MLJBase.check(model::TMLEstimator, args... ; full=false) = true

Base.merge(ndt₁::AbstractNode, ndt₂::AbstractNode) = 
    node((ndt₁, ndt₂) -> merge(ndt₁, ndt₂), ndt₁, ndt₂)

fluctuation_input(covariate, offset) = (covariate=covariate, offset=offset)
fluctuation_input(covariate::AbstractNode, offset::AbstractNode) =
    node((c, o) -> fluctuation_input(c, o), covariate, offset)

"""

Adapts the type of the treatment variable passed to the G learner
"""
adapt(T::NamedTuple{<:Any, NTuple{1, Z}}) where Z = T[1]
adapt(T) = T
adapt(T::AbstractNode) = node(adapt, T)

###############################################################################
## Reporting utilities
###############################################################################

function queryreport(br; tail=:both)
    fitres = br.fitresult
    pval = pvalue(fitres.estimate, fitres.stderror, tail=tail)
    confint = confinterval(fitres.estimate, fitres.stderror)
    (pvalue=pval, confint=confint, fitres...)
end
queryreport(br::Vector; tail=:both) =
    [queryreport(x, tail=tail) for x in br]

    """
    pvalue(m::Machine{TMLEstimator})

Computes the p-value associated with the estimated quantity.
"""
function pvalue(estimate, stderror; tail=:both)
    x = estimate/stderror

    dist = Normal(0, 1)
    if tail == :both
        min(2 * min(cdf(dist, x), ccdf(dist, x)), 1.0)
    elseif tail == :left
        cdf(dist, x)
    elseif tail == :right
        ccdf(dist, x)
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end

"""
    confinterval(m::Machine{TMLEstimator})

Provides a 95% confidence interval for the true quantity of interest.
"""
function confinterval(estimate, stderror)
    return (estimate - 1.96stderror, estimate + 1.96stderror)
end

###############################################################################
## Interactions Generation
###############################################################################

"""
    interaction_combinations(query::NamedTuple{names,})
Returns a generator over the different combinations of interactions that
can be built from a query.
"""
function interaction_combinations(query::NamedTuple{names,}) where names
    return (NamedTuple{names}(query) for query in Iterators.product(query...))
end


"""
    indicator_fns(query::NamedTuple{names,})

Implements the (-1)^{n-j} formula representing the cross-value of
indicator functions,  where:
    - n is the order of interaction considered
    - j is the number of treatment variables different from the "case" value
"""
function indicator_fns(query::NamedTuple{nms,}) where nms
    case = NamedTuple{nms}([v[1] for v in query])
    interactionorder = length(query)
    return Dict(q => (-1)^(interactionorder - sum(q[key] == case[key] for key in nms)) 
                for q in interaction_combinations(query))
end


###############################################################################
## Offset
###############################################################################

target_prob(y) = y.prob_given_ref[2]
target_prob(y::AbstractNode) = node(y->target_prob(y), y)

function compute_offset(Q̅mach::Machine{<:Probabilistic}, X)
    # The machine is an estimate of a probability distribution
    # In the binary case, the expectation is assumed to be the probability of the second class
    ŷ = MLJ.predict(Q̅mach, X)
    expectation = target_prob(ŷ)
    return logit(expectation)
end


function compute_offset(Q̅mach::Machine{<:Deterministic}, X)
    return MLJ.predict(Q̅mach, X)
end

###############################################################################
## Covariate
###############################################################################

function indicator_values(indicators, T)
    covariate = zeros(nrows(T))
    for (i, row) in enumerate(Tables.namedtupleiterator(T))
        if haskey(indicators, row)
            covariate[i] = indicators[row]
        end
    end
    covariate
end
indicator_values(indicators, T::AbstractNode) = 
    node(t -> indicator_values(indicators, t), T)


function plateau_likelihood(likelihood, threshold, verbosity)
    likelihood = max.(threshold, likelihood)
    # Log indices for which p(T|W) < threshold as this indicates very rare events.
    if verbosity > 0
        idx_under_threshold = findall(x -> x <= threshold, likelihood)
        length(idx_under_threshold) > 0 && @info "p(T|W) evaluated under $threshold at indices: $idx_under_threshold"
    end
    likelihood
end

plateau_likelihood(likelihood::AbstractNode, threshold, verbosity) = 
    node(l -> plateau_likelihood(l, threshold, verbosity), likelihood)

elemwise_divide(x, y) = x ./ y
elemwise_divide(x::AbstractNode, y::AbstractNode) = 
    node((x, y) -> elemwise_divide(x,y), x, y)

"""
For each data point, computes: (-1)^(interaction-oder - j)
Where j is the number of treatments different from the reference in the query.
"""
function compute_covariate(Gmach::Machine, W, T, indicators; verbosity=1, threshold=0.005)
    # Compute the indicator value
    indic_vals = indicator_values(indicators, T)

    # Compute density and truncate
    likelihood = density(Gmach, W, T)

    likelihood = plateau_likelihood(likelihood, threshold, verbosity)
    
    return elemwise_divide(indic_vals, likelihood)
end


###############################################################################
## Fluctuation
###############################################################################

init_counterfactual_fluctuation(T::AbstractNode) =
    node(t -> zeros(nrows(t)), T)


function counterfactualTreatment(vals, T)
    names = keys(vals)
    n = nrows(T)
    NamedTuple{names}(
            [categorical(repeat([vals[name]], n), levels=levels(Tables.getcolumn(T, name)))
                            for name in names])
end
counterfactualTreatment(vals, T::AbstractNode) =
    node(t -> counterfactualTreatment(vals, t), T)


function compute_fluctuation(Fmach::Machine, 
                             Q̅mach::Machine, 
                             Gmach::Machine, 
                             Hmach::Machine,
                             indicators,
                             W, 
                             T; 
                             verbosity=1,
                             threshold=0.005)
    Thot = transform(Hmach, T)
    X = merge(Thot, W)
    offset = compute_offset(Q̅mach, X)
    covariate = compute_covariate(Gmach, W, T, indicators; 
                                    verbosity=verbosity,
                                    threshold=threshold)
    Xfluct = fluctuation_input(covariate, offset)
    return  MLJ.predict_mean(Fmach, Xfluct)
end

"""
    counterfactual_fluctuations(query, 
                                Fmach,
                                Q̅mach,
                                Gmach,
                                Hmach,
                                W,
                                T)
                                
Computes the Counterfactual value of the fluctuation.
If the order of Interaction is 2 with binary variables, this is:
 1/n ∑ [ Fluctuation(t₁=1, t₂=1, W=w) - Fluctuation(t₁=1, t₂=0, W=w)
        - Fluctuation(t₁=0, t₂=1, W=w) + Fluctuation(t₁=0, t₂=0, W=w)]
"""
function counterfactual_fluctuations(Fmach::Machine,
                                     Q̅mach::Machine,
                                     Gmach::Machine,
                                     Hmach::Machine,
                                     indicators,
                                     W,
                                     T; 
                                     verbosity=1,
                                     threshold=0.005)

    ct_fluct = init_counterfactual_fluctuation(T)
    for (vals, sign) in indicators 
        counterfactualT = counterfactualTreatment(vals, T)
        ct_fluct += sign*compute_fluctuation(Fmach, 
                                             Q̅mach, 
                                             Gmach, 
                                             Hmach,
                                             indicators,
                                             W, 
                                             counterfactualT; 
                                             verbosity=verbosity,
                                             threshold=threshold)
    end
    return ct_fluct
end
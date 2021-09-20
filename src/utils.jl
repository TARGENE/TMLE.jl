logit(X) = log.(X ./ (1 .- X))
expit(X) = 1 ./ (1 .+ exp.(-X))

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


idx_under_threshold(d, t) = findall

"""

Adapts the type of the treatment variable passed to the G learner
"""
adapt(T::NamedTuple{<:Any, NTuple{1, Z}}) where Z = T[1]
adapt(T) = T

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
## Offset and covariate
###############################################################################

function compute_offset(Q̅mach::Machine{<:Probabilistic}, X)
    # The machine is an estimate of a probability distribution
    # In the binary case, the expectation is assumed to be the probability of the second class
    expectation = MLJ.predict(Q̅mach, X).prob_given_ref[2]
    return logit(expectation)
end


function compute_offset(Q̅mach::Machine{<:Deterministic}, X)
    return MLJ.predict(Q̅mach, X)
end


"""
For each data point, computes: (-1)^(interaction-oder - j)
Where j is the number of treatments different from the reference in the query.
"""
function compute_covariate(Gmach::Machine, W, T, query; verbosity=1)
    threshold = 0.005
    # Build the Indicator function dictionary
    indicators = indicator_fns(query)
    
    # Compute the indicator value
    covariate = zeros(nrows(T))
    for (i, row) in enumerate(Tables.namedtupleiterator(T))
        if haskey(indicators, row)
            covariate[i] = indicators[row]
        end
    end

    # Compute density and truncate
    d = density(Gmach, W, T)

    # Log indices for which p(T|W) < threshold as this indicates very rare events.
    d = max.(threshold, d)
    if verbosity > 0
        idx_under_threshold = findall(x -> x <= threshold, d)
        length(idx_under_threshold) > 0 && @info "p(T|W) evaluated under $threshold at indices: $idx_under_threshold"
    end
    
    return covariate ./ d
end


###############################################################################
## Fluctuation
###############################################################################

function compute_fluctuation(Fmach::Machine, 
                             Q̅mach::Machine, 
                             Gmach::Machine, 
                             Hmach::Machine,
                             W, 
                             T; 
                             verbosity=1)
    Thot = transform(Hmach, T)
    X = merge(Thot, W)
    offset = compute_offset(Q̅mach, X)
    cov = compute_covariate(Gmach, W, T, Fmach.model.query; verbosity=verbosity)
    Xfluct = (covariate=cov, offset=offset)
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
function counterfactual_fluctuations(query, 
                                     Fmach,
                                     Q̅mach,
                                     Gmach,
                                     Hmach,
                                     W,
                                     T; 
                                     verbosity=1)
    indicators = indicator_fns(query)
    n = nrows(T)
    ct_fluct = zeros(n)
    for (ct, sign) in indicators 
        names = keys(ct)
        counterfactualT = NamedTuple{names}(
            [categorical(repeat([ct[name]], n), levels=levels(Tables.getcolumn(T, name)))
                            for name in names])
        ct_fluct += sign*compute_fluctuation(Fmach, 
                                Q̅mach, 
                                Gmach, 
                                Hmach,
                                W, 
                                counterfactualT; 
                                verbosity=verbosity)
    end
    return ct_fluct
end
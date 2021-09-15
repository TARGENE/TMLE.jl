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

function compute_offset(y_cond_exp_estimate::Machine{<:Probabilistic}, X)
    # The machine is an estimate of a probability distribution
    # In the binary case, the expectation is assumed to be the probability of the second class
    expectation = MLJ.predict(y_cond_exp_estimate, X).prob_given_ref[2]
    return logit(expectation)
end


function compute_offset(y_cond_exp_estimate::Machine{<:Deterministic}, X)
    return MLJ.predict(y_cond_exp_estimate, X)
end


"""
For each data point, computes: (-1)^(interaction-oder - j)
Where j is the number of treatments different from the reference in the query.
"""
function compute_covariate(Gmach::Machine, W, T, query)
    # Build the Indicator function dictionary
    indicators = indicator_fns(query)
    
    # Compute the indicator value
    Trow = Tables.rowtable(T)
    covariate = zeros(nrows(T))
    for (i, row) in enumerate(Tables.rows(Trow))
        if haskey(indicators, row)
            covariate[i] = indicators[row]
        end
    end

    # Compute density and truncate
    d = density(Gmach, W, Tables.matrix(T))
    # is this really necessary/suitable?
    d = min.(0.995, max.(0.005, d))
    return covariate ./ d
end
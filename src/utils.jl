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
Computes: ∏ᵢ (2tᵢ - 1) / likelihood
Which seems to cover all cases covered so far with a joint categorical treatment.
"""
function compute_covariate(t_likelihood_estimate::Machine, W, T, t_target)
    # tpred is an estimate of a probability distribution
    # we need to extract the observed likelihood out of it
    res = ones(size(t_target)[1])
    for colname in Tables.columnnames(T)
        res .*= (2Tables.getcolumn(T, colname) .- 1)
    end

    tpred = MLJ.predict(t_likelihood_estimate, W)
    likelihood = pdf.(tpred, t_target)
    # truncate predictions, is this really necessary/suitable?
    likelihood = min.(0.995, max.(0.005, likelihood))
    return res ./ likelihood
end
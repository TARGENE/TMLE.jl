
"""
    ATEEstimator(Q̅, 
                G,
                fluctuation_family)

# Scope:

Implements the Targeted Minimum Loss-Based Estimator for the Average Treatment Effect (ATE).
The Average Treatment Effect is defined as: ATE = E[E[Y|T=1, W=w] - E[Y|T=0, W=w]],
where:

- Y is the target variable (Binary)
- T is the treatment variable (Binary)
- W are confounder variables

The TMLE procedure relies on plugin estimation. Here, the ATE requires an 
estimator of t,w → E[Y|T=t, W=w], an estimator of w → p(T|w) and an estimator
of w → p(w). The empirical distribution will be used for w → p(w) all along. 
The estimator of t,w → E[Y|T=t, W=w] is then fluctuated to solve the efficient influence
curve equation. 

More information can be found about TMLE in "Causal Inference for Observational and Experimental Data"
by Mark J. van der Laan and Sherri Rose.

# Arguments:

- Q̅::MLJ.Supervised : The learner to be used
for E[Y|W, T]. Typically a `MLJ.Stack`.
- G::MLJ.Supervised : The learner to be used
for p(T|W). Typically a `MLJ.Stack`.
- fluctuation_family::Distribution : This will be used to build the fluctuation 
using a GeneralizedLinearModel. Typically `Normal` for a continuous target 
and `Bernoulli` for a Binary target.

# Examples:

TODO

"""
mutable struct ATEEstimator <: TMLEstimator 
    Q̅::MLJ.Supervised
    G::MLJ.Supervised
    fluctuation_family::Distribution
end



###############################################################################
## Fluctuation
###############################################################################

function compute_fluctuation(fitted_fluctuator::GeneralizedLinearModel, 
                         Q̅mach::Machine, 
                         QGmach::Machine, 
                         W, 
                         t_target)
    T = (t=float(t_target),)
    X = merge(T, W)
    offset = compute_offset(Q̅mach, X)
    cov = compute_covariate(QGmach, W, T, t_target)
    return  GLM.predict(fitted_fluctuator, reshape(cov, :, 1); offset=offset)
end

###############################################################################
## Fit
###############################################################################


"""
    MLJ.fit(tmle::ATEEstimator, 
            verbosity::Int, 
            t::CategoricalVector{Bool}, 
            W, 
            y::Union{CategoricalVector{Bool}, Vector{<:Real}})
"""
function MLJ.fit(tmle::ATEEstimator, 
             verbosity::Int, 
             t::CategoricalVector{Bool}, 
             W, 
             y::Union{CategoricalVector{Bool}, Vector{<:Real}})
    n = nrows(y)

    # Convert to NamedTuples
    W = Tables.columntable(W)
    T = (t=float(t),)
    X = merge(T, W)
    t_target = t

    # Initial estimate of E[Y|A, W]
    Q̅mach = machine(tmle.Q̅, X, y)
    fit!(Q̅mach, verbosity=verbosity)

    # Estimate of P(A|W)
    QGmach = machine(tmle.G, W, t_target)
    fit!(QGmach, verbosity=verbosity)

    # Fluctuate E[Y|A, W] 
    # on the covariate and the offset 
    offset = compute_offset(Q̅mach, X)
    covariate = compute_covariate(QGmach, W, T, t_target)
    fluctuator = glm(reshape(covariate, :, 1), y, tmle.fluctuation_family; offset=offset)

    # Compute the final estimate tmleATE = 1/n ∑ Fluctuator(t=1, W=w) - Fluctuator(t=0, W=w)
    fluct = (compute_fluctuation(fluctuator, 
                                Q̅mach, 
                                QGmach, 
                                W, 
                                categorical(ones(Bool, n), levels=levels(t_target)))
            - compute_fluctuation(fluctuator, 
                                Q̅mach, 
                                QGmach,
                                W, 
                                categorical(zeros(Bool, n), levels=levels(t_target))))

    estimate = mean(fluct)

    # Standard error from the influence curve
    observed_fluct = GLM.predict(fluctuator, reshape(covariate, n, 1); offset=offset)
    inf_curve = covariate .* (float(y) .- observed_fluct) .+ fluct .- estimate

    fitresult = (
        estimate=estimate,
        stderror=sqrt(var(inf_curve)/n),
        mean_inf_curve=mean(inf_curve),
        Q̅mach=Q̅mach,
        QGmach=QGmach,
        fluctuation=fluctuator
        )

    cache = nothing
    report = nothing
    return fitresult, cache, report

end

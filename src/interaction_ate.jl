
"""
    InteractionATEEstimator(Q̅,
                            G,
                            fluctuation_family)

# Scope:

Implements the Targeted Minimum Loss-Based Estimator for the Interaction 
Average Treatment Effect (IATE) defined by Beentjes and Khamseh in
https://link.aps.org/doi/10.1103/PhysRevE.102.053314.
For instance, The IATE is defined for two treatment variables as: 

IATE = E[E[Y|T₁=1, T₂=1, W=w] - E[E[Y|T₁=1, T₂=0, W=w]
        - E[E[Y|T₁=0, T₂=1, W=w] + E[E[Y|T₁=0, T₂=0, W=w]

where:

- Y is the target variable (Binary)
- T = T₁, T₂ are the treatment variables (Binary)
- W are confounder variables

The TMLE procedure relies on plugin estimation. Like the ATE, the IATE 
requires an estimator of t,w → E[Y|T=t, W=w], an estimator of  w → p(T|w) 
and an estimator of w → p(w). The empirical distribution will be used for w → p(w) all along. 
The estimator of t,w → E[Y|T=t, W=w] is then fluctuated to solve the efficient influence
curve equation. 

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
mutable struct InteractionATEEstimator <: TMLEstimator 
    Q̅::MLJ.Supervised
    G::MLJ.Supervised
    fluctuation::Fluctuation
end


###############################################################################
## Fit
###############################################################################


"""
    MLJ.fit(tmle::InteractionATEEstimator, 
                 verbosity::Int, 
                 T,
                 W, 
                 y::Union{CategoricalVector{Bool}, Vector{<:Real}}
"""
function MLJ.fit(tmle::InteractionATEEstimator, 
                 verbosity::Int, 
                 T,
                 W, 
                 y::Union{CategoricalVector{Bool}, Vector{<:Real}})
    n = nrows(y)
    Tnames = [t for t in Tables.columnnames(T)]

    # Initial estimate of E[Y|T, W]:
    #   - The treatment variables are hot-encoded  
    #   - W and T are merged
    #   - The machine is implicitely fit
    # (Maybe check T and X don't have the same column names?)
    Hmach = machine(OneHotEncoder(features=Tnames, drop_last=true), T)
    fit!(Hmach, verbosity=verbosity)
    Thot = transform(Hmach, T)

    W = Tables.columntable(W)

    X = merge(Thot, W)
    
    Q̅mach = machine(tmle.Q̅, X, y)
    fit!(Q̅mach, verbosity=verbosity)

    # Initial estimate of P(T|W)
    #   - T is converted to an Array
    #   - The machine is implicitely fit
    Gmach = machine(tmle.G, W, T)
    fit!(Gmach, verbosity=verbosity)

    # Fluctuate E[Y|T, W] 
    # on the covariate and the offset 
    offset = compute_offset(Q̅mach, X)
    covariate = compute_covariate(Gmach, W, T, tmle.fluctuation.query)
    Xfluct = (covariate=covariate, offset=offset)
    
    Fmach = machine(tmle.fluctuation, Xfluct, y)
    fit!(Fmach, verbosity=verbosity)

    # Compute the final estimate 
    ct_fluct = counterfactual_fluctuations(tmle.fluctuation.query, 
                                     Fmach,
                                     Q̅mach,
                                     Gmach,
                                     Hmach,
                                     W,
                                     T)

    estimate = mean(ct_fluct)

    # Standard error from the influence curve
    observed_fluct = MLJ.predict_mean(Fmach, Xfluct)
    inf_curve = covariate .* (float(y) .- observed_fluct) .+ ct_fluct .- estimate

    fitresult = (
    estimate=estimate,
    stderror=sqrt(var(inf_curve)/n),
    mean_inf_curve=mean(inf_curve),
    Q̅mach=Q̅mach,
    Gmach=Gmach,
    Fmach=Fmach
    )

    cache = nothing
    report = nothing
    return fitresult, cache, report
end

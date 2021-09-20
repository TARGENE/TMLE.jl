"""
    TMLEstimator(Q̅, G, F)

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

The TMLEstimator procedure relies on plugin estimation. Like the ATE, the IATE 
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
mutable struct TMLEstimator <: MLJ.Model 
    Q̅::MLJ.Supervised
    G::MLJ.Supervised
    fluctuation::Fluctuation
end



"""

Let's default to no warnings for now.
"""
MLJBase.check(model::TMLEstimator, args... ; full=false) = true

pvalue(tmle::TMLEstimator, estimate, stderror) = 2*(1 - cdf(Normal(0, 1), abs(estimate/stderror)))

confint(tmle::TMLEstimator, estimate, stderror) = (estimate - 1.96stderror, estimate + 1.96stderror)

###############################################################################
## Fit
###############################################################################


"""
    MLJ.fit(tmle::TMLEstimator, 
                 verbosity::Int, 
                 T,
                 W, 
                 y::Union{CategoricalVector{Bool}, Vector{<:Real}}
"""
function MLJ.fit(tmle::TMLEstimator, 
                 verbosity::Int, 
                 T,
                 W, 
                 y::Union{CategoricalVector{Bool}, Vector{<:Real}})
    # Converting all tables to NamedTuples
    T = Tables.columntable(T)
    W = Tables.columntable(W)
    intersect(keys(T), keys(W)) == [] || throw("T and W should have different column names")

    # Initial estimate of E[Y|T, W]:
    #   - The treatment variables are hot-encoded  
    #   - W and T are merged
    #   - The machine is implicitely fit
    Hmach = machine(OneHotEncoder(drop_last=true), T)
    fit!(Hmach, verbosity=verbosity)
    Thot = transform(Hmach, T)

    X = merge(Thot, W)
    Q̅mach = machine(tmle.Q̅, X, y)
    fit!(Q̅mach, verbosity=verbosity)

    # Initial estimate of P(T|W)
    #   - T is converted to an Array
    #   - The machine is implicitely fit
    Gmach = machine(tmle.G, W, adapt(T))
    fit!(Gmach, verbosity=verbosity)

    # Fluctuate E[Y|T, W] 
    # on the covariate and the offset 
    offset = compute_offset(Q̅mach, X)
    covariate = compute_covariate(Gmach, W, T, tmle.fluctuation.query; verbosity=verbosity)
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
                                     T;
                                     verbosity=verbosity)

    estimate = mean(ct_fluct)

    # Standard error from the influence curve
    observed_fluct = MLJ.predict_mean(Fmach, Xfluct)
    inf_curve = covariate .* (float(y) .- observed_fluct) .+ ct_fluct .- estimate

    fitresult = (
    estimate=estimate,
    stderror=sqrt(var(inf_curve)/nrows(y)),
    mean_inf_curve=mean(inf_curve),
    Q̅mach=Q̅mach,
    Gmach=Gmach,
    Fmach=Fmach
    )

    cache = nothing
    report = nothing
    return fitresult, cache, report
end

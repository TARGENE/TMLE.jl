mutable struct TMLEstimator <: MLJ.DeterministicComposite 
    Q̅::MLJ.Supervised
    G::MLJ.Supervised
    F::Fluctuation
    R::Report
    threshold::Float64
end

"""
    TMLEstimator(Q̅, G, F, query; threshold=0.005)

Implements the Targeted Minimum Loss-Based Estimator introduced by
van der Laan in https://pubmed.ncbi.nlm.nih.gov/22611591/. Two functionals of the 
data generating distribution can currently be estimated:

- The classic Average Treatment Effect (ATE)
- The Interaction Average Treatment Effect (IATE) defined by Beentjes and Khamseh in
https://link.aps.org/doi/10.1103/PhysRevE.102.053314. For instance, The IATE is defined for two treatment variables as: 

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

- Q̅: A Supervised learner for E[Y|W, T]
- G: A Supervised learner for p(T | W)
- F: A Fluctuation, see continuousfluctuation, binaryfluctuation


- threshold: p(T | W) is truncated to this value to avoid division overflows.
"""
TMLEstimator(Q̅, G, F; threshold=0.005) = 
    TMLEstimator(Q̅, G, F, Report(), threshold)


"""

Let's default to no warnings for now.
"""
MLJBase.check(model::TMLEstimator, args... ; full=false) = true

"""
    briefreport(m::Machine{TMLEstimator})

Returns the reported results, see Report.
"""
briefreport(m::Machine{TMLEstimator}) = fitted_params(m).R.fitresult

"""
    Distributions.estimate(m::Machine{TMLEstimator})

Returns the estimated quantity from a fitted machines.
"""
Distributions.estimate(m::Machine{TMLEstimator}) = briefreport(m).estimate

"""
    Distributions.stderror(m::Machine{TMLEstimator})

Returns the standard error associated with the estimate from a fitted machines. 
"""
Distributions.stderror(m::Machine{TMLEstimator}) = briefreport(m).stderror

"""
    pvalue(m::Machine{TMLEstimator})

Computes the p-value associated with the estimated quantity.
"""
function pvalue(m::Machine{TMLEstimator}; tail=:both)
    res = briefreport(m)
    x = res.estimate/res.stderror

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
function confinterval(m::Machine{TMLEstimator})
    res = briefreport(m)
    return (res.estimate - 1.96res.stderror, res.estimate + 1.96res.stderror)
end

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
    Ts = source(T)
    Ws = source(W)
    ys = source(y)

    # Converting all tables to NamedTuples
    T = node(t->NamedTuple{keys(tmle.F.query)}(Tables.columntable(t)), Ts)
    W = node(w->Tables.columntable(w), Ws)
    # intersect(keys(T), keys(W)) == [] || throw("T and W should have different column names")

    # Initial estimate of E[Y|T, W]:
    #   - The treatment variables are hot-encoded  
    #   - W and T are merged
    #   - The machine is implicitely fit
    Hmach = machine(OneHotEncoder(drop_last=true), T)
    Thot = transform(Hmach, T)

    X = node((t, w) -> merge(t, w), Thot, W)
    Q̅mach = machine(tmle.Q̅, X, ys)

    # Initial estimate of P(T|W)
    #   - T is converted to an Array
    #   - The machine is implicitely fit
    Gmach = machine(tmle.G, W, adapt(T))

    # Fluctuate E[Y|T, W] 
    # on the covariate and the offset 
    offset = compute_offset(Q̅mach, X)
    covariate = compute_covariate(Gmach, W, T, tmle.F.indicators; 
                                    verbosity=verbosity,
                                    threshold=tmle.threshold)
    Xfluct = fluctuation_input(covariate, offset)

    Fmach = machine(tmle.F, Xfluct, ys)

    # Compute the final estimate 
    ct_fluct = counterfactual_fluctuations(Fmach,
                                           Q̅mach,
                                           Gmach,
                                           Hmach,
                                           W,
                                           T;
                                           verbosity=verbosity,
                                           threshold=tmle.threshold)

    # Fit the Report
    observed_fluct = MLJ.predict_mean(Fmach, Xfluct)

    Rmach = machine(tmle.R, ct_fluct, observed_fluct, covariate, ys)
    out = MLJ.predict(Rmach, ct_fluct)

    mach = machine(Deterministic(), Ts, Ws, ys; predict=out)

    return!(mach, tmle, verbosity)
end



mutable struct TMLEstimator <: DeterministicComposite 
    Q̅::Supervised
    G::Supervised
    F::Union{LinearRegressor, LinearBinaryClassifier}
    queries::Tuple{Vararg{Query}}
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
- queries...: At least one query
- threshold: p(T | W) is truncated to this value to avoid division overflows.
"""
function TMLEstimator(Q̅::Supervised, G::Supervised, queries::Vararg{Query}; threshold=0.005::Float64)
    if Q̅ isa Probabilistic
        F = LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)
    elseif Q̅ isa Deterministic
        F = LinearRegressor(fit_intercept=false, offsetcol = :offset)
    end
    TMLEstimator(Q̅, G, F, queries, threshold)
end


###############################################################################
## Fit
###############################################################################

"""
    MLJBase.fit(tmle::TMLEstimator, 
                 verbosity::Int, 
                 T,
                 W, 
                 y::Union{CategoricalVector{Bool}, Vector{<:Real}}

As per all MLJ inputs, T and W should respect the Tables.jl interface.
"""
function MLJBase.fit(tmle::TMLEstimator, 
                 verbosity::Int, 
                 T,
                 W, 
                 Y)

    check_ordering(tmle.queries, T)
    
    Ts = source(T)
    Ws = source(W)
    Ys = source(Y)

    # Filtering missing values before fit
    T, W = TableOperations.dropmissing(Ts, Ws)

    # Initial estimate of P(T|W)
    Gmach = machine(tmle.G, W, adapt(T))

    Hmach = machine(OneHotEncoder(drop_last=true), T)
    Thot = transform(Hmach, T)

    X = node((t, w) -> merge(t, w), Thot, W)

    reported = []
    predicted = []
    extreme_propensity = nothing
    # Loop over targets, an estimator is fit for each target
    for (target_idx, target_name) in enumerate(Tables.columnnames(Y))
        ys = Tables.getcolumn(Ys, target_name)
        # Initial estimate of E[Y|T, W]:
        Q̅mach = machine(tmle.Q̅, X, ys)

        offset = compute_offset(Q̅mach, X)
        # Loop over queries that will define new covariate values
        for (query_idx, query) in enumerate(tmle.queries)
            indicators = indicator_fns(query)
            covariate = compute_covariate(Gmach, W, T, indicators; 
                                        threshold=tmle.threshold)
            # Log extreme values
            extreme_propensity = log_over_threshold(covariate, tmle.threshold)

            # Fluctuate E[Y|T, W] 
            # on the covariate and the offset 
            Xfluct = fluctuation_input(covariate, offset)
            Fmach = machine(tmle.F, Xfluct, ys)
            
            observed_fluct = predict_mean(Fmach, Xfluct)

            queryreport = estimation_report(Fmach,
                            Q̅mach,
                            Gmach,
                            Hmach,
                            W,
                            T,
                            observed_fluct,
                            ys,
                            covariate,
                            indicators,
                            tmle.threshold,
                            query,
                            target_name)
            report_key = queryreportname(target_idx, query_idx)
            push!(reported, NamedTuple{(report_key,)}([queryreport]))
            # This is actually empty but required
            push!(predicted, observed_fluct)
        end
    end

    predicted = hcat(predicted...)

    mach = machine(Deterministic(), Ts, Ws, Ys; 
            predict=predicted, 
            report=(extreme_propensity_idx=extreme_propensity, merge(reported...)...)
    )
    return!(mach, tmle, verbosity)
end


###############################################################################
## Complementary methods
###############################################################################

MLJBase.reformat(::TMLEstimator, T, W, Y) = (T, W, totable(Y))
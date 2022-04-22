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
                 Y)

Estimates the Average Treatment Effect or the Interaction Average Treatment Effect 
using the TMLE framework.

# Arguments:
    - T: A table representing treatment variables. If multiple treatments are provided,
    the interaction effect (IATE) is estimated.
    - W: A table of confounding variables.
    - Y: A vector or a table. If Y is a table, p(T|W) is fit only once and E[Y|T,W] 
    is fit for each column in Y. If the number of target variables in large, it helps 
    to drastically reduce the computational time.
"""
function MLJBase.fit(tmle::TMLEstimator, T, W, Y; 
    verbosity::Int=1, 
    cache=false,
    callbacks=[MachineReportBuilder()])

    T, W, Y = MLJBase.reformat(tmle::TMLEstimator, T, W, Y)
    # Fitting the encoder
    Hmach = machine(OneHotEncoder(drop_last=true), T, cache=cache)
    fit_with_callbacks!(Hmach, callbacks, verbosity, :Encoder)

    # Fitting P(T|W)
    Gmach = machine(tmle.G, W, adapt(T), cache=cache)
    fit_with_callbacks!(Gmach, callbacks, verbosity, :G)

    queryreports = NamedTuple{}()
    extreme_propensity = nothing
    # Loop over targets, an estimator is fit for each target
    for (target_idx, target_name) in enumerate(Tables.columnnames(Y))
        # Get the target as a table
        y = Tables.columntable(TableOperations.select(Y, target_name))
        # Filter missing values from tables
        T_, W_, y_ = TableOperations.dropmissing(T, W, y)
        
        # Thot is a Floating point representation of T
        # y_ is a vector
        Thot_ = transform(Hmach, T_)
        y_ = first(y_)

        # Fitting E[Y|T, W]
        X = merge(Thot_, W_)
        Q̅mach = machine(tmle.Q̅, X, y_, cache=cache)
        fit_with_callbacks!(Q̅mach, callbacks, verbosity, Symbol(:Q, '_', target_idx))

        offset = compute_offset(Q̅mach, X)
        # Loop over queries that will define new covariate values
        for (query_idx, query) in enumerate(tmle.queries)
            indicators = indicator_fns(query)
            covariate = compute_covariate(Gmach, W_, T_, indicators; 
                                        threshold=tmle.threshold)
            # Log extreme values
            extreme_propensity = log_over_threshold(covariate, tmle.threshold)

            # Fluctuate E[Y|T, W] 
            # on the covariate and the offset 
            Xfluct = fluctuation_input(covariate, offset)
            Fmach = machine(tmle.F, Xfluct, y_, cache=cache)
            fit_with_callbacks!(Fmach, callbacks, verbosity, Symbol(:Q, '_', target_idx, '_', query_idx))

            observed_fluct = predict_mean(Fmach, Xfluct)

            queryreport = estimation_report(Fmach,
                            Q̅mach,
                            Gmach,
                            Hmach,
                            W_,
                            T_,
                            observed_fluct,
                            y_,
                            covariate,
                            indicators,
                            tmle.threshold,
                            query,
                            target_name)
            report_key = queryreportname(target_idx, query_idx)
            queryreports = merge(queryreports, NamedTuple{(report_key,)}([queryreport]))
            # This is actually empty but required
        end
    end
    estimation_report_ = (queryreports=queryreports,)
    return finalize_with_callbacks!(estimation_report_, callbacks)
end


###############################################################################
## Complementary methods
###############################################################################

function fit_with_callbacks!(mach, callbacks, verbosity, id)
    fit!(mach, verbosity=verbosity)
    for callback in callbacks
        after_machine_fit(callback, mach, id)
    end
end

function finalize_with_callbacks!(estimation_report::NamedTuple, callbacks)
    for callback in callbacks
        estimation_report = finalize(estimation_report, callback)
    end
    return estimation_report
end

function check_columnnames(T, W, Y)
    Tnames = Tables.columnnames(T)
    Wnames = Tables.columnnames(W)
    Ynames = Tables.columnnames(Y)

    combinations = [(("T", Tnames), ("W", Wnames)), 
                    (("T", Tnames), ("Y", Ynames)),
                    (("W", Wnames), ("Y", Ynames))]
    for ((input₁, colnames₁), (input₂, colnames₂)) in combinations
        columns_intersection = intersect(colnames₁, colnames₂)
        if length(columns_intersection) != 0
            throw(ArgumentError(string(input₁, " and ", input₂, " share some column names:", columns_intersection)))
        end
    end
end

function MLJBase.reformat(tmle::TMLEstimator, T, W, Y)
    Y = totable(Y)
    check_columnnames(T, W, Y)
    check_ordering(tmle.queries, T)
   return (T, W, Y)
end
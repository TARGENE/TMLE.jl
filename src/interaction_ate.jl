
"""
    InteractionATEEstimator(target_cond_expectation_estimator,
                            treatment_cond_likelihood_estimator,
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

- target_cond_expectation_estimator::MLJ.Supervised : The learner to be used
for E[Y|W, T]. Typically a `MLJ.Stack`.
- treatment_cond_likelihood_estimator::MLJ.Supervised : The learner to be used
for p(T|W). Typically a `MLJ.Stack`.
- fluctuation_family::Distribution : This will be used to build the fluctuation 
using a GeneralizedLinearModel. Typically `Normal` for a continuous target 
and `Bernoulli` for a Binary target.

# Examples:

TODO
"""
mutable struct InteractionATEEstimator <: TMLEstimator 
    target_cond_expectation_estimator::MLJ.Supervised
    treatment_cond_likelihood_estimator::MLJ.Supervised
    fluctuation_family::Distribution
end

function tomultivariate(T)
    t = []
    mapping = Dict((true, true)=>1, (true, false)=>2, (false, true)=>3, (false, false)=>4)
    features = Tables.schema(T).names
    for row in Tables.rows(T)
        push!(t, mapping[(row[features[1]], row[features[2]])])
    end
    return categorical(t)
end



###############################################################################
## Fluctuation
###############################################################################

function compute_fluctuation(fitted_fluctuator::GeneralizedLinearModel, 
                            target_expectation_mach::Machine, 
                            treatment_likelihood_mach::Machine, 
                            W, 
                            T, 
                            t_target)
    X = merge(T, W)
    offset = compute_offset(target_expectation_mach, X)
    cov = compute_covariate(treatment_likelihood_mach, W, T, t_target)
    return  GLM.predict(fitted_fluctuator, reshape(cov, :, 1); offset=offset)
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

    # Get T as a target and convert to float so that it will not be hot encoded
    # by the target_expectation_mach
    t_target = tomultivariate(T)
    Tnames = Tables.columnnames(T)
    T = NamedTuple{Tnames}([float(Tables.getcolumn(T, colname)) for colname in Tnames])

    # Maybe check T and X don't have the same column names?
    W = Tables.columntable(W)
    X = merge(T, W)
    
    # Initial estimate of E[Y|A, W]
    target_expectation_mach = machine(tmle.target_cond_expectation_estimator, X, y)
    fit!(target_expectation_mach, verbosity=verbosity)

    # Estimate of P(A|W)
    treatment_likelihood_mach = machine(tmle.treatment_cond_likelihood_estimator, W, t_target)
    fit!(treatment_likelihood_mach, verbosity=verbosity)

    # Fluctuate E[Y|A, W] 
    # on the covariate and the offset 
    offset = compute_offset(target_expectation_mach, X)
    covariate = compute_covariate(treatment_likelihood_mach, W, T, t_target)
    fluctuator = glm(reshape(covariate, :, 1), y, tmle.fluctuation_family; offset=offset)

    # Compute the final estimate 
    # InteractionATE = 1/n ∑ [ Fluctuator(t₁=1, t₂=1, W=w) - Fluctuator(t₁=1, t₂=0, W=w)
    #                        - Fluctuator(t₁=0, t₂=1, W=w) + Fluctuator(t₁=0, t₂=0, W=w)]
    counterfactual_treatments = [(ones, ones, 1), 
                                 (ones, zeros, -1), 
                                 (zeros, ones, -1), 
                                 (zeros, zeros, 1)]
    fluct = zeros(n)
    for (t₁, t₂, sign) in counterfactual_treatments
        counterfactualT = NamedTuple{Tnames}([t₁(n), t₂(n)])
        counterfactual_t_target = tomultivariate(counterfactualT)
        fluct .+= sign*compute_fluctuation(fluctuator, 
                                target_expectation_mach, 
                                treatment_likelihood_mach, 
                                W, 
                                counterfactualT,
                                counterfactual_t_target)
    end

    estimate = mean(fluct)

    # Standard error from the influence curve
    observed_fluct = GLM.predict(fluctuator, reshape(covariate, n, 1); offset=offset)
    inf_curve = covariate .* (float(y) .- observed_fluct) .+ fluct .- estimate

    fitresult = (
    estimate=estimate,
    stderror=sqrt(var(inf_curve)/n),
    mean_inf_curve=mean(inf_curve),
    target_expectation_mach=target_expectation_mach,
    treatment_likelihood_mach=treatment_likelihood_mach,
    fluctuation=fluctuator
    )

    cache = nothing
    report = nothing
    return fitresult, cache, report
end
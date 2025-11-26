import GLMNet

"""
        LassoCTMLE <: CollaborativeStrategy

LASSO-based Collaborative TMLE strategy for high-dimensional causal inference.

# Notes
- Confounders are automatically extracted from the provided `estimand` at runtime
    (via `extract_confounders_from_estimand(Ψ)`). The constructor no longer requires
    an explicit `confounders` argument; callers may still build custom propensity
    specifications by calling `propensity_score(Ψ, confounders_list, strategy)`.
- Uses GLMNet cross-validation to select the optimal lambda automatically.
- No refitting: coefficients from the CV fit are reused directly for efficiency.

# Parameters
- `cv_folds`: Number of cross-validation folds for lambda selection
- `alpha`: Elastic Net mixing parameter (1.0 = LASSO, 0.0 = Ridge)

# Example
```julia
strategy = LassoCTMLE(cv_folds = 5, alpha = 1.0)
estimator = Tmle(collaborative_strategy = strategy)
result, _ = estimator(estimand, data)
```
"""
mutable struct LassoCTMLE <: CollaborativeStrategy
    cv_folds::Int
    alpha::Float64
    initial_fit::Any
    used::Bool

    function LassoCTMLE(; 
        cv_folds = 5,
        alpha = 1.0
    )
        new(cv_folds, alpha, nothing, false)
    end
end

# helper for conditional logs
log_info(strategy::LassoCTMLE, msg) = @debug msg

function fit_glmnet_propensity_score(var_names, strategy::LassoCTMLE)
    # use CV-selected optimal lambda from the stored fit
    if strategy.initial_fit === nothing
        throw(ErrorException("LassoCTMLE requires a GLMNet CV fit stored in `strategy.initial_fit`. Ensure the strategy has been initialized."))
    end
    
    cv_fit = strategy.initial_fit
    path = cv_fit.path
    
    # use optimal lambda from CV (minimum mean loss)
    optimal_lambda_idx = argmin(cv_fit.meanloss)
    optimal_lambda = cv_fit.lambda[optimal_lambda_idx]
    idx = optimal_lambda_idx
    
    coeffs = path.betas[:, idx]
    selected_indices = findall(x -> abs(x) > 1e-6, coeffs)
    
    if isempty(selected_indices)
        @warn "No variables selected by GLMNet at optimal λ=$optimal_lambda, using all variables"
        return var_names, cv_fit, idx
    end
    
    selected_vars = var_names[selected_indices]
    return selected_vars, cv_fit, idx
end

"""
Extract a vector of confounder symbols from the estimand `Ψ`.
Collects treatment-specific confounders (in order) and returns unique symbols.
"""
function extract_confounders_from_estimand(Ψ)
    Ψtreatments = TMLE.treatments(Ψ)
    all = Symbol[]
    for T in Ψtreatments
        if hasproperty(Ψ, :treatment_confounders) && haskey(Ψ.treatment_confounders, T)
            append!(all, collect(Ψ.treatment_confounders[T]))
        end
    end
    return unique(all)
end

function initialise!(strategy::LassoCTMLE, Ψ)
    strategy.used = false
    return nothing
end

update!(strategy::LassoCTMLE, g, ĝ) = nothing

finalise!(strategy::LassoCTMLE) = nothing

function exhausted(strategy::LassoCTMLE)
    # strategy runs once with CV-optimal lambda, then is exhausted
    return strategy.used
end

"""
Create propensity score specification using the given confounders list.
"""
function propensity_score(Ψ, confounders_list::Vector{Symbol}, strategy::LassoCTMLE)
    Ψtreatments = TMLE.treatments(Ψ)
    return Tuple(map(eachindex(Ψtreatments)) do index
        T = Ψtreatments[index]
        T_confounders = intersect(confounders_list, Ψ.treatment_confounders[T])
        T_parents = (T_confounders..., Ψtreatments[index+1:end]...)
        TMLE.ConditionalDistribution(T, T_parents)
    end)
end

"""
Get propensity score specification from the collaborative strategy.
"""
function propensity_score(Ψ, strategy::LassoCTMLE)
    confounders = extract_confounders_from_estimand(Ψ)
    return propensity_score(Ψ, confounders, strategy)
end

"""
Iterator implementation for LASSO-based collaborative TMLE.
Runs once with GLMNet CV-selected optimal lambda.
"""
function Base.iterate(it::TMLE.StepKPropensityScoreIterator{LassoCTMLE})
    strategy = it.collaborative_strategy
    
    # only run once
    if strategy.used
        return nothing
    end
    
    # extract confounders once (used throughout)
    confounders = extract_confounders_from_estimand(it.Ψ)
    
    # run GLMNet CV if not already done
    if strategy.initial_fit === nothing
        treatment_var = first(TMLE.treatments(it.Ψ))
        y_binary = Int.(unwrap.(it.dataset[!, treatment_var]))
        confounder_data = it.dataset[!, confounders]
        X_matrix = Matrix{Float64}(confounder_data)
        
        # run CV to get optimal lambda
        strategy.initial_fit = GLMNet.glmnetcv(X_matrix, y_binary, alpha=strategy.alpha, nfolds=strategy.cv_folds)
        # find lambda with minimum CV loss
        optimal_lambda_idx = argmin(strategy.initial_fit.meanloss)
        optimal_lambda = strategy.initial_fit.lambda[optimal_lambda_idx]
        log_info(strategy, "LassoCTMLE: CV selected λ=$optimal_lambda")
    end
    
    # get variable selection from CV fit
    selected_confounders, glm_fit, lambda_idx = fit_glmnet_propensity_score(confounders, strategy)
    
    log_info(strategy, "LassoCTMLE: Selected $(length(selected_confounders))/$(length(confounders)) confounders")
    
    # build propensity score specification
    g = propensity_score(it.Ψ, selected_confounders, strategy)
    
    # build prefit estimator using CV coefficients (no refitting)
    path = glm_fit.path
    selected_indices = [findfirst(==(v), confounders) for v in selected_confounders]
    coeffs_full = path.betas[:, lambda_idx]
    coeffs_selected = coeffs_full[selected_indices]
    intercept = path.a0[lambda_idx]
    
    components = Dict{Symbol, Tuple}()
    for cd in g
        components[cd.outcome] = (selected_confounders, coeffs_selected, intercept)
    end
    ĝ = TMLE.PrefitGLMNetJointConditionalDistributionEstimator(components)
    
    strategy.used = true
    
    # return optimal lambda (computed once above or from cached fit)
    optimal_lambda_idx = argmin(glm_fit.meanloss)
    optimal_lambda = glm_fit.lambda[optimal_lambda_idx]
    
    return (g, ĝ), optimal_lambda
end

Base.iterate(it::TMLE.StepKPropensityScoreIterator{LassoCTMLE}, state) = nothing

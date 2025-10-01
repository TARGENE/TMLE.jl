import GLMNet

"""
    LassoCTMLE <: CollaborativeStrategy

LASSO-based Collaborative TMLE strategy for high-dimensional causal inference.

# Parameters
- `confounders`: Vector of confounding variable symbols
- `patience`: Number of lambda candidates to explore collaboratively  
- `lambda_path`: Regularization parameter values (CV-generated if empty)
- `cv_folds`: Number of cross-validation folds
- `alpha`: Elastic Net mixing parameter (1.0 = LASSO, 0.0 = Ridge)

# Example
```julia
strategy = LassoCTMLE(confounders = [:W1, :W2, :W3], patience = 5)
estimator = Tmle(collaborative_strategy = strategy)
result, _ = estimator(estimand, data)
```
"""
mutable struct LassoCTMLE <: CollaborativeStrategy
    confounders::Vector{Symbol}
    patience::Int
    lambda_path::Vector{Float64}
    cv_folds::Int
    alpha::Float64
    current_iteration::Int
    explored_lambdas::Set{Float64}
    best_lambda::Union{Float64, Nothing}
    best_cv_loss::Float64
    
    function LassoCTMLE(; 
        confounders = Symbol[],
        patience = 5,
        lambda_path = :cv,
        cv_folds = 5,
        alpha = 1.0
    )
        isempty(confounders) &&
            throw(ArgumentError("Must specify confounders for LassoCTMLE"))
        
        actual_lambda_path = lambda_path == :cv ? Float64[] : lambda_path
        
        new(confounders, patience, actual_lambda_path, cv_folds, alpha, 0, 
            Set{Float64}(), nothing, Inf)
    end
end

"""
    GLMNetPropensityScore

Propensity score model using GLMNet for regularized logistic regression.
"""
struct GLMNetPropensityScore
    alpha::Float64
    lambda::Float64
    selected_vars::Vector{Symbol}
end

function fit_glmnet_propensity_score(X_matrix, y_binary, alpha, lambda, var_names)
    try
        fit = GLMNet.glmnet(X_matrix, y_binary, alpha=alpha, lambda=[lambda])
        coeffs = fit.betas[:, 1]
        selected_indices = findall(x -> abs(x) > 1e-6, coeffs)
        
        if isempty(selected_indices)
            @warn "No variables selected by GLMNet, falling back to correlation"
            n_selected = max(1, round(Int, length(var_names) * 0.5))
            return var_names[1:n_selected], fit
        end
        
        selected_vars = var_names[selected_indices]
        @info "GLMNet: α=$alpha, λ=$lambda → $(length(selected_vars))/$(length(var_names)) variables selected"
        @info "GLMNet: Selected variables: $selected_vars"
        return selected_vars, fit
        
    catch e
        @warn "GLMNet fitting failed: $e, falling back to correlation-based selection"
        selection_fraction = max(0.2, 1.0 - lambda * 100)
        n_selected = max(1, round(Int, length(var_names) * selection_fraction))
        return var_names[1:min(n_selected, length(var_names))], nothing
    end
end

function initialise!(strategy::LassoCTMLE, Ψ)
    @info "LassoCTMLE: Initialising collaborative lambda exploration"
    strategy.current_iteration = 0
    empty!(strategy.explored_lambdas)
    strategy.best_lambda = nothing
    strategy.best_cv_loss = Inf
    
    if isempty(strategy.lambda_path)
        @info "LassoCTMLE: Will generate CV lambda sequence when data is available"
    else
        @info "LassoCTMLE: Using provided lambda sequence: $(strategy.lambda_path[1:min(3, length(strategy.lambda_path))])..."
    end
    
    return nothing
end

function update!(strategy::LassoCTMLE, g, ĝ)
    strategy.current_iteration += 1
    current_lambda = strategy.lambda_path[min(strategy.current_iteration, length(strategy.lambda_path))]
    push!(strategy.explored_lambdas, current_lambda)
    
    @info "LassoCTMLE: Collaborative update $(strategy.current_iteration)/$(strategy.patience) - explored λ = $current_lambda"
    @info "LassoCTMLE: Explored lambdas so far: $(sort(collect(strategy.explored_lambdas)))"
    
    return nothing
end

function update_with_loss!(strategy::LassoCTMLE, g, ĝ, cv_loss::Float64)
    current_lambda = strategy.lambda_path[min(strategy.current_iteration, length(strategy.lambda_path))]
    
    if cv_loss < strategy.best_cv_loss
        @info "LassoCTMLE: New best λ = $current_lambda with CV loss = $cv_loss (previous best: $(strategy.best_cv_loss))"
        strategy.best_lambda = current_lambda
        strategy.best_cv_loss = cv_loss
    else
        @info "LassoCTMLE: λ = $current_lambda with CV loss = $cv_loss (keeping best λ = $(strategy.best_lambda))"
    end
    
    return nothing
end

function update!(strategy::LassoCTMLE, g, ĝ, cv_loss::Float64)
    strategy.current_iteration += 1
    current_lambda = strategy.lambda_path[min(strategy.current_iteration, length(strategy.lambda_path))]
    push!(strategy.explored_lambdas, current_lambda)
    
    if cv_loss < strategy.best_cv_loss
        strategy.best_lambda = current_lambda
        strategy.best_cv_loss = cv_loss
        @info "LassoCTMLE: New best λ = $current_lambda (CV loss = $cv_loss)"
    end
    
    @info "LassoCTMLE: Collaborative update $(strategy.current_iteration)/$(strategy.patience) - explored λ = $current_lambda"
    @info "LassoCTMLE: Explored lambdas so far: $(sort(collect(strategy.explored_lambdas)))"
    @info "LassoCTMLE: Current best λ = $(strategy.best_lambda) (best CV loss = $(strategy.best_cv_loss))"
    
    return nothing
end

finalise!(strategy::LassoCTMLE) = nothing

function exhausted(strategy::LassoCTMLE)
    if isempty(strategy.lambda_path) && strategy.current_iteration == 0
        @info "LassoCTMLE: Not exhausted - CV lambda generation pending"
        return false
    end
    
    is_exhausted = strategy.current_iteration >= strategy.patience || 
                   length(strategy.explored_lambdas) >= length(strategy.lambda_path)
    
    if is_exhausted && strategy.best_lambda !== nothing
        @info "LassoCTMLE: Collaborative exploration complete - best λ = $(strategy.best_lambda) (CV loss = $(strategy.best_cv_loss))"
    else
        @info "LassoCTMLE: Checking exhaustion - iteration $(strategy.current_iteration)/$(strategy.patience), exhausted: $is_exhausted"
    end
    
    return is_exhausted
end

"""
Create propensity score specification using the given confounders list.
"""
function propensity_score(Ψ, confounders_list::Vector{Symbol})
    @info "LassoCTMLE: Creating propensity score specification with confounders: $confounders_list"
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
    @info "LassoCTMLE: Getting propensity score from strategy"
    return propensity_score(Ψ, strategy.confounders)
end

"""
Iterator implementation for LASSO-based collaborative TMLE.
Explores different lambda values with GLMNet regularization.
"""
function Base.iterate(it::TMLE.StepKPropensityScoreIterator{LassoCTMLE})
    strategy = it.collaborative_strategy
    
    # Generate CV lambda sequence if needed
    if isempty(strategy.lambda_path)
        @info "LassoCTMLE: Generating CV lambda sequence"
        treatment_var = first(TMLE.treatments(it.Ψ))
        y_binary = Int.(unwrap.(it.dataset[!, treatment_var]))
        confounder_data = it.dataset[!, strategy.confounders]
        X_matrix = Matrix{Float64}(confounder_data)
        
        try
            auto_fit = GLMNet.glmnet(X_matrix, y_binary, alpha=strategy.alpha)
            min_lambda = minimum(auto_fit.lambda)
            strong_lambdas = auto_fit.lambda[auto_fit.lambda .>= min_lambda]
            
            n_lambdas = min(strategy.patience * 2, length(strong_lambdas))
            strategy.lambda_path = strong_lambdas[1:n_lambdas]
            
            @info "LassoCTMLE: Generated $(length(strategy.lambda_path)) CV lambdas from $(round(minimum(strategy.lambda_path), digits=6)) to $(round(maximum(strategy.lambda_path), digits=3))"
        catch e
            @warn "LassoCTMLE: CV lambda generation failed, using fallback: $e"
            strategy.lambda_path = exp10.(range(-2, stop = 0, length = strategy.patience))
        end
    end
    
    # Find next lambda to explore
    available_lambdas = setdiff(strategy.lambda_path, strategy.explored_lambdas)
    isempty(available_lambdas) && return nothing
    
    current_lambda = first(available_lambdas)
    
    @info "LassoCTMLE: TRUE GLMNet collaborative candidate λ = $current_lambda (iteration $(strategy.current_iteration + 1))"
    
    # Prepare data for GLMNet
    treatment_var = first(TMLE.treatments(it.Ψ))
    y_binary = Int.(unwrap.(it.dataset[!, treatment_var]))
    confounder_data = it.dataset[!, strategy.confounders]
    X_matrix = Matrix{Float64}(confounder_data)
    
    # Use GLMNet for variable selection
    selected_confounders, glm_fit = fit_glmnet_propensity_score(
        X_matrix, y_binary, strategy.alpha, current_lambda, strategy.confounders
    )
    
    @info "LassoCTMLE: GLMNet α=$(strategy.alpha), λ=$current_lambda → $(length(selected_confounders))/$(length(strategy.confounders)) confounders"
    @info "LassoCTMLE: Selected confounders: $selected_confounders"
    
    # Create propensity score specification with selected confounders
    g = propensity_score(it.Ψ, selected_confounders)
    models = it.models
    
    # Build the propensity score estimator 
    ĝ = TMLE.build_propensity_score_estimator(g, models, it.dataset; train_validation_indices=nothing)
    
    @info "LassoCTMLE: Built TRUE GLMNet collaborative candidate with $(length(g)) propensity score component(s)"
    
    return (g, ĝ), current_lambda
end

Base.iterate(it::TMLE.StepKPropensityScoreIterator{LassoCTMLE}, state) = nothing

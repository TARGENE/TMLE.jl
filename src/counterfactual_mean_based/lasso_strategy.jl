##################################################################
###                      LassoStrategy                         ###
##################################################################

using GLMNet
using DataFrames
using StatsFuns
using ..TMLE
using Statistics

"""
    LassoCTMLE <: CollaborativeStrategy

Lasso-based Collaborative TMLE strategy using cross-validated lambda selection.
"""
struct LassoCTMLE <: CollaborativeStrategy
    lambda_path::Vector{Float64}
    cv_folds::Int
    best_lambda::Union{Nothing,Float64}
    losses::Vector{Float64}
    confounders::Vector{Symbol}
    patience::Int
    function LassoCTMLE(;
        lambda_path = exp10.(range(-4, stop = 0, length = 50)),
        cv_folds = 5,
        confounders = Symbol[],
        patience = 3,
    )
        isempty(confounders) &&
            throw(ArgumentError("Must specify confounders for LassoCTMLE"))
        new(lambda_path, cv_folds, nothing, Float64[], confounders, patience)
    end
end

"""
    stratified_kfold(y::AbstractVector, k::Int)

Returns a vector of (train_idx, test_idx) tuples for stratified k-fold cross-validation,
where `y` contains class labels.
"""
function stratified_kfold(y::AbstractVector, k::Int)
    idx_by_class = Dict{eltype(y), Vector{Int}}()
    for (i, label) in enumerate(y)
        push!(get!(idx_by_class, label, Int[]), i)
    end

    folds = [Int[] for _ in 1:k]
    for idxs in values(idx_by_class)
        shuffled = shuffle(idxs)
        for (i, idx) in enumerate(shuffled)
            push!(folds[mod1(i, k)], idx)
        end
    end

    result = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, k)
    all_idxs = collect(1:length(y))
    for i in 1:k
        test_idx = folds[i]
        train_idx = setdiff(all_idxs, test_idx)
        result[i] = (train_idx, test_idx)
    end
    return result
end

function initialise!(strategy::LassoCTMLE, Ψ)
    strategy.best_lambda = nothing
    empty!(strategy.losses)
    return nothing
end

function update!(strategy::LassoCTMLE, last_targeted_η̂ₙ, dataset)
    return nothing
end

finalise!(strategy::LassoCTMLE) = nothing

function exhausted(strategy::LassoCTMLE)
    strategy.best_lambda !== nothing
end

"""
    propensity_score(Ψ, strategy::LassoCTMLE)

Returns a function which, given a dataset, fits LASSO at the chosen lambda and predicts propensity scores.
This method is generic: it works for any valid Ψ parameter by using TMLE.treatment(Ψ).
"""
function propensity_score(Ψ, strategy::LassoCTMLE)
    function fit_predict(dataset; lambda_override = nothing)
        W = Matrix(select(dataset, strategy.confounders))
        A = dataset[!, treatment(Ψ)]
        λ = isnothing(lambda_override) ? [strategy.best_lambda] : [lambda_override]
        g_fit = glmnet(W, A, Binomial(), lambda = λ)
        g_pred = GLMNet.predict(g_fit, W, lambda = λ[1])
        clamp.(g_pred, 0.01, 0.99) # stabilize
    end
    return fit_predict
end

"""
    crossvalidate_lambda(strategy, Ψ, dataset, cv_folds)

Selects the best lambda from the path via cross-validated log-loss.
"""
function crossvalidate_lambda(strategy::LassoCTMLE, Ψ, dataset, cv_folds)
    W = Matrix(select(dataset, strategy.confounders))
    A = dataset[!, treatment(Ψ)]
    folds = stratified_kfold(A, cv_folds)
    losses = zeros(length(strategy.lambda_path))
    for (train_idx, val_idx) in folds
        W_train, W_val = W[train_idx, :], W[val_idx, :]
        A_train, A_val = A[train_idx], A[val_idx]
        g_fit = glmnet(W_train, A_train, Binomial(), lambda = strategy.lambda_path)
        for (i, λ) in enumerate(g_fit.lambda)
            g_pred = GLMNet.predict(g_fit, W_val, lambda = λ)
            g_pred = clamp.(g_pred, 0.01, 0.99)
            # Log-loss
            losses[i] += -mean(A_val .* log.(g_pred) .+ (1 .- A_val) .* log.(1 .- g_pred))
        end
    end
    avg_losses = losses ./ cv_folds
    best_idx = argmin(avg_losses)
    return strategy.lambda_path[best_idx], avg_losses
end

"""
    LassoCTMLEIterator

Once best lambda is chosen, this iterator provides a single candidate fit at that lambda.
"""
struct LassoCTMLEIterator
    strategy::LassoCTMLE
    Ψ::Any
    dataset::Any
    models::Any
    last_targeted_η̂ₙ::Any
end

function Base.iterate(it::LassoCTMLEIterator, state = 1)
    if state > 1
        return nothing
    end
    strategy, Ψ, dataset = it.strategy, it.Ψ, it.dataset
    W = Matrix(select(dataset, strategy.confounders))
    A = dataset[!, treatment(Ψ)]
    λ = strategy.best_lambda
    g_fit = glmnet(W, A, Binomial(), lambda = [λ])
    ĝ =
        (g, dataset; kwargs...) -> begin
            g_pred = GLMNet.predict(
                g,
                Matrix(select(dataset, strategy.confounders)),
                lambda = λ,
            )
            clamp.(g_pred, 0.01, 0.99)
        end
    return ((g_fit, ĝ), state+1)
end

"""
    step_k_best_candidate(
        collaborative_strategy::LassoCTMLE,
        Ψ, dataset, models, fluctuation_model, last_targeted_η̂ₙ, last_loss;
        ...kwargs...
    )

Pipeline step: cross-validates lambda, sets best_lambda, fits candidate at best lambda, and targets using TMLE fluctuation machinery.
NOTE: This strategy does NOT compute clever covariate or target the parameter directly.
      It only provides a g-model fit (propensity score model) using LASSO.
      TMLE.jl machinery handles clever covariate and targeting for any valid Ψ.
"""
function step_k_best_candidate(
    collaborative_strategy::LassoCTMLE,
    Ψ,
    dataset,
    models,
    fluctuation_model,
    last_targeted_η̂ₙ,
    last_loss;
    verbosity = 1,
    cache = Dict(),
    machine_cache = false,
    acceleration = CPU1(),
)

    best_lambda, avg_losses = crossvalidate_lambda(
        collaborative_strategy,
        Ψ,
        dataset,
        collaborative_strategy.cv_folds,
    )
    collaborative_strategy.best_lambda = best_lambda
    collaborative_strategy.losses = avg_losses

    iterator =
        LassoCTMLEIterator(collaborative_strategy, Ψ, dataset, models, last_targeted_η̂ₙ)
    (g, ĝ), _ = iterate(iterator, 1)

    ĝₙ = ĝ(g, dataset)
    targeted_η̂ₙ, loss = TMLE.get_new_targeted_candidate(
        last_targeted_η̂ₙ,
        ĝₙ,
        fluctuation_model,
        dataset;
        verbosity = verbosity-1,
        cache = cache,
        machine_cache = machine_cache,
    )

    return g, ĝ, targeted_η̂ₙ, loss, false
end


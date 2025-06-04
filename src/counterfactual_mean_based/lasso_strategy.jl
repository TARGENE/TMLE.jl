##################################################################
###                      LassoStrategy                            ###
#####################################################################

using GLMNet
using DataFrames
using StatsFuns
using MLBase: StratifiedKFold

"""
    LassoCTMLE <: CollaborativeStrategy

Lasso-based Collaborative TMLE strategy using cross-validated lambda selection.
"""
struct LassoCTMLE <: CollaborativeStrategy
    lambda_path::Vector{Float64}
    cv_folds::Int
    best_lambda::Union{Nothing, Float64}
    losses::Vector{Float64}
    confounders::Vector{Symbol}
    
    function LassoCTMLE(; 
        lambda_path=exp10.(range(-4, stop=0, length=50)), 
        cv_folds=5, 
        confounders=Symbol[]
    )
        isempty(confounders) && throw(ArgumentError("Must specify confounders for LassoCTMLE"))
        new(lambda_path, cv_folds, nothing, Float64[], confounders)
    end
end

function initialise!(strategy::LassoCTMLE, Ψ)
    strategy.best_lambda = nothing
    empty!(strategy.losses)
    return nothing
end

function update!(strategy::LassoCTMLE, g, ĝ)
    
    return nothing
end

finalise!(strategy::LassoCTMLE) = nothing

function exhausted(strategy::LassoCTMLE)
    strategy.best_lambda !== nothing
end

struct LassoCTMLEIterator
    strategy::LassoCTMLE
    Ψ
    dataset
    models
end

function Base.iterate(it::LassoCTMLEIterator)
    W = Matrix(select(it.dataset, it.strategy.confounders))
    A = it.dataset[!, treatment(it.Ψ)]
    Y = it.dataset[!, outcome(it.Ψ)]
    n = length(Y)
    
    folds = StratifiedKFold(A, it.strategy.cv_folds)
    lambda_losses = zeros(length(it.strategy.lambda_path))
    
    for (fold, (train_idx, val_idx)) in enumerate(folds)
        g_fit = glmnet(W[train_idx, :], A[train_idx], Binomial(), lambda=it.strategy.lambda_path)
        for (i, λ) in enumerate(g_fit.lambda)
            g_pred = GLMNet.predict(g_fit, W[val_idx, :], lambda=λ)
            g_pred = clamp.(g_pred, 0.01, 0.99)
            Q_pred = it.models.Q_pred[val_idx]  # Use models from initial Q fit
            H = A[val_idx] ./ g_pred .- (1 .- A[val_idx]) ./ (1 .- g_pred)
            Q_star = fluctuate(Q_pred, H, Y[val_idx])
            lambda_losses[i] += mean(-Y[val_idx].*log.(Q_star) .- (1 .- Y[val_idx]).*log.(1 .- Q_star))
        end
    end
    
    avg_losses = lambda_losses ./ it.strategy.cv_folds
    best_idx = argmin(avg_losses)
    it.strategy.best_lambda = it.strategy.lambda_path[best_idx]
    it.strategy.losses = avg_losses
    
    return (build_final_estimator(it), nothing)
end

function build_final_estimator(it)
    W = Matrix(select(it.dataset, it.strategy.confounders))
    A = it.dataset[!, treatment(it.Ψ)]
    
    g_final = glmnet(W, A, Binomial(), lambda=[it.strategy.best_lambda])
    g_pred = GLMNet.predict(g_final, W)
    g_pred = clamp.(g_pred, 0.01, 0.99)
    
    
    H = A ./ g_pred .- (1 .- A) ./ (1 .- g_pred)
    Q_star = fluctuate(it.models.Q_pred, H, it.dataset[!, outcome(it.Ψ)])
    
    
    ψ = ATE(Q_star, A, it.dataset[!, outcome(it.Ψ)])
    return TMLEResult(
        ψ=ψ.ate,
        SE=ψ.se,
        CI=(ψ.ci_lower, ψ.ci_upper),
        pvalue=ψ.pvalue
    )
end
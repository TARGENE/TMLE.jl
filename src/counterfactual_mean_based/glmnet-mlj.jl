import MLJBase
import GLMNet

mutable struct GLMNetRegressor <: MLJBase.Deterministic
    resampling::MLJBase.ResamplingStrategy
    params::Dict
end

"""
    GLMNetRegressor(;resampling=CV(), params...)

A GLMNet regressor for continuous outcomes based on the `glmnetcv` function from the [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) 
package.

# Arguments:

- resampling: A MLJ `ResamplingStrategy`, see [MLJ resampling strategies](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/#Built-in-resampling-strategies)
- params: Additional parameters to the `glmnetcv` function

# Examples:

A glmnet with `alpha=0`.

```julia

model = GLMNetRegressor(resampling=CV(nfolds=3), alpha=0)
mach = machine(model, X, y)
fit!(mach, verbosity=0)
```
"""
GLMNetRegressor(;resampling=MLJBase.CV(), params...) = GLMNetRegressor(resampling, Dict(params))

mutable struct GLMNetClassifier <: MLJBase.Probabilistic
    resampling::MLJBase.ResamplingStrategy
    params::Dict
end

"""
    GLMNetClassifier(;resampling=StratifiedCV(), params...)

A GLMNet classifier for binary/multinomial outcomes based on the `glmnetcv` function from the [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) 
package.

# Arguments:

- resampling: A MLJ `ResamplingStrategy`, see [MLJ resampling strategies](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/#Built-in-resampling-strategies)
- params: Additional parameters to the `glmnetcv` function

# Examples:

A glmnet with `alpha=0`.

```julia

model = GLMNetClassifier(resampling=StratifiedCV(nfolds=3), alpha=0)
mach = machine(model, X, y)
fit!(mach, verbosity=0)
```
"""
GLMNetClassifier(;resampling=MLJBase.StratifiedCV(), params...) = GLMNetClassifier(resampling, Dict(params))

GLMNetModel = Union{GLMNetRegressor, GLMNetClassifier}

make_fitresult(::GLMNetRegressor, res, y) = (glmnetcv=res, )
make_fitresult(::GLMNetClassifier, res, y) = (glmnetcv=res, levels=sort(unique(y)))

function getfolds(resampling, X, y)
    n = size(y, 1)
    folds = Vector{Int}(undef, n)
    for (split_index, (_, val_indices)) in enumerate(MLJBase.train_test_pairs(resampling, 1:n, X, y))
        folds[val_indices] .= split_index
    end
    return folds
end

function MLJBase.fit(model::GLMNetModel, verbosity::Int, X, y)
    folds = getfolds(model.resampling, X, y)
    res = GLMNet.glmnetcv(MLJBase.matrix(X), y; folds=folds, model.params...)
    # This is currently not caught by the GLMNet package
    if length(res.meanloss) == 0
        throw(error("glmnetcv's mean loss is empty. Probably meaning convergence failed at the first lambda for some fold."))
    end
    return make_fitresult(model, res, y), nothing, nothing
end

MLJBase.predict(::GLMNetRegressor, fitresult, X) =
    GLMNet.predict(fitresult.glmnetcv, MLJBase.matrix(X))

function MLJBase.predict(::GLMNetClassifier, fitresult, X)
    raw_probs = GLMNet.predict(fitresult.glmnetcv, MLJBase.matrix(X), outtype=:prob)
    levels = fitresult.levels
    if size(levels, 1) == 2
        probs = hcat(1 .- raw_probs, raw_probs)
        preds = MLJBase.UnivariateFinite(levels, probs, pool=missing)
    else
        preds = MLJBase.UnivariateFinite(levels, raw_probs, pool=missing)
    end
    return preds
end

MLJBase.input_scitype(::Type{<:GLMNetModel}) = MLJBase.Table{<:AbstractVector{<:MLJBase.Continuous}}
MLJBase.target_scitype(::Type{<:GLMNetRegressor}) = AbstractVector{<:MLJBase.Continuous}
MLJBase.target_scitype(::Type{<:GLMNetClassifier}) = AbstractVector{<:MLJBase.Finite}

